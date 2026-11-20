"""Role operations must not depend on where they are written, and must fail cleanly.

Covers the pieces the main role tests leave out: one reference model of how roles
should end up, checked at every position; regression tests for the position,
``roles``-copy and target-dtype bugs; and a seeded random sweep that fits many
small recipes and checks the result against the model.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import pandas as pd
import polars as pl
import pytest
from sklearn.base import clone
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Recipe, StepFilter, StepNormalize
from mlfeatures import Selectors as s
from mlfeatures.compat import RecipeTransformer
from mlfeatures.types import Roles

COLUMNS = ["x1", "x2", "x3", "y"]


def _as_dict(roles: Roles) -> dict[str, set[str]]:
    return {c: roles.of(c) for c in roles.all_columns() if roles.of(c)}


def _polars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [4.0, 3.0, 2.0, 1.0],
            "x3": [2.0, 1.0, 4.0, 3.0],
            "y": [0, 1, 0, 1],
        }
    )


def _pandas() -> pd.DataFrame:
    return _polars().to_pandas()


# A step that only touches x2, so x1 and x3 stay untouched by every recipe below.
def _step() -> StepNormalize:
    return StepNormalize(["x2"])


# ---------------------------------------------------------------------------
# The reference model: defaults are given once, then each operation edits roles.
# ---------------------------------------------------------------------------

Op = tuple[str, str, str | None]  # (kind, column, role or old_role)


def _model(ops: list[tuple[str, str, str, str | None]]) -> dict[str, set[str]] | None:
    """Expected roles after ``ops`` on a recipe whose target is ``y``.

    Each op is ``(kind, column, role, old_role)``. Returns ``None`` when the
    documented rules say fitting raises.
    """
    roles: dict[str, set[str]] = {c: {"feature"} for c in COLUMNS if c != "y"}
    roles["y"] = {"target"}
    implicit = {c for c in COLUMNS if c != "y"}
    for kind, column, role, old in ops:
        held = roles[column]
        if kind == "remove":
            if role not in held:
                return None
            held.discard(role)
            if role == "feature":
                implicit.discard(column)
            continue
        if kind == "update":
            gone = {old} if old is not None else set(held)
            if not gone <= held:
                return None
            held -= gone
            if "feature" in gone:
                implicit.discard(column)
        if role in ("target", "case_weights") and column in implicit:
            held.discard("feature")
            implicit.discard(column)
        elif role == "feature":
            implicit.discard(column)
        held.add(role)
        if {"feature", "target"} <= held:
            return None
    return {c: h for c, h in roles.items() if h}


def _build(
    ops: list[tuple[str, str, str, str | None]],
    *,
    steps_before: int,
    data: Any,
) -> Recipe:
    """The recipe with every op written after ``steps_before`` steps."""
    rec = Recipe(data=data, target="y")
    for _ in range(steps_before):
        rec = rec.add(_step())
    for kind, column, role, old in ops:
        if kind == "add":
            rec = rec.add_role(column, role=role)
        elif kind == "remove":
            rec = rec.remove_role(column, role=role)
        else:
            rec = rec.update_role(column, new_role=role, old_role=old)
    return rec


SINGLE_OPS: list[tuple[str, str, str, str | None]] = [
    ("add", "x1", "target", None),
    ("add", "x1", "case_weights", None),
    ("add", "x1", "group", None),
    ("add", "x1", "feature", None),
    ("add", "y", "feature", None),
    ("remove", "x1", "feature", None),
    ("remove", "x1", "group", None),
    ("update", "x1", "target", None),
    ("update", "x1", "target", "feature"),
    ("update", "x1", "case_weights", "feature"),
    ("update", "x1", "id", None),
]


class TestPositionInvariance:
    @pytest.mark.parametrize("op", SINGLE_OPS, ids=lambda o: "-".join(map(str, o)))
    @pytest.mark.parametrize("steps_before", [0, 1, 3])
    @pytest.mark.parametrize("make", [_polars, _pandas], ids=["polars", "pandas"])
    def test_one_op_matches_the_model_at_every_position(
        self,
        op: tuple[str, str, str, str | None],
        steps_before: int,
        make: Callable[[], Any],
    ) -> None:
        expected = _model([op])
        rec = _build([op], steps_before=steps_before, data=make())
        if expected is None:
            with pytest.raises(ValueError):
                rec.fit(make())
        else:
            assert _as_dict(rec.fit(make()).roles) == expected

    @pytest.mark.parametrize("steps_before", [0, 2])
    def test_two_ops_on_one_column_match_the_model(self, steps_before: int) -> None:
        for first in SINGLE_OPS:
            for second in SINGLE_OPS:
                ops = [first, second]
                expected = _model(ops)
                rec = _build(ops, steps_before=steps_before, data=_polars())
                if expected is None:
                    with pytest.raises(ValueError):
                        rec.fit(_polars())
                else:
                    assert _as_dict(rec.fit(_polars()).roles) == expected, ops

    @pytest.mark.parametrize("where", ["before", "after_filter", "after_empty"])
    def test_target_on_a_default_feature_around_row_and_identity_steps(
        self, where: str
    ) -> None:
        rec = Recipe(data=_polars(), target="y")
        if where == "after_filter":
            rec = rec.add(StepFilter(lambda frame: frame["x1"] > 1))
        elif where == "after_empty":
            rec = rec.add(StepNormalize(s.has_role("nothing")))
        fitted = rec.add_role("x3", role="target").fit(_polars())
        assert fitted.roles.of("x3") == {"target"}


class TestRolesProperty:
    def test_mutating_raises_without_leading_operations(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = Recipe(data=_polars(), target="y")
        with pytest.raises(TypeError) as excinfo:
            rec.roles.add("x1", "id")
        assert str(excinfo.value) == snapshot
        with pytest.raises(TypeError):
            rec.roles.remove("y", "target")
        assert rec.fit(_polars()).roles.of("x1") == {"feature"}

    def test_mutating_raises_with_leading_operations(self) -> None:
        rec = Recipe(data=_polars(), target="y").add_role("x2", role="group")
        with pytest.raises(TypeError):
            rec.roles.add("x1", "id")
        assert rec.roles.of("x1") == set()
        assert rec.roles.of("x2") == {"group"}

    def test_copy_of_roles_is_mutable(self) -> None:
        rec = Recipe(data=_polars(), target="y")
        mutable = rec.roles.copy()
        mutable.add("x1", "id")
        assert mutable.of("x1") == {"id"}
        assert rec.roles.of("x1") == set()

    def test_is_read_only(self) -> None:
        rec = Recipe(data=_polars(), target="y")
        with pytest.raises(AttributeError):
            rec.roles = rec.roles  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]

    def test_set_roles_after_an_operation_shows_in_roles(self) -> None:
        rec = (
            Recipe(data=_polars(), target="y")
            .update_role("x1", new_role="id")
            .set_roles(group="x1")
        )
        assert rec.roles.of("x1") == {"id", "group"}
        reverse = (
            Recipe(data=_polars(), target="y")
            .set_roles(group="x1")
            .update_role("x1", new_role="id")
        )
        assert reverse.roles.of("x1") == {"id"}


class TestRequirementsAcrossPositions:
    def test_role_less_column_is_still_required_at_transform(self) -> None:
        # A column left with no role is kept, and an unroled column defaults to
        # required, so transform() still needs it.
        fitted = (
            Recipe(data=_polars(), target="y")
            .remove_role("x1", role="feature")
            .add(_step())
            .fit(_polars())
        )
        assert fitted.roles.of("x1") == set()
        assert fitted.schema is not None
        assert fitted.schema.transform_required["x1"] is True
        with pytest.raises(ValueError):
            fitted.transform(_polars().drop("x1", "y"))

    def test_late_operation_in_a_selector_override_is_not_required(self) -> None:
        fitted = (
            Recipe(data=_polars(), target="y", features=["x2"])
            .update_role_requirements("id", transform_required=False)
            .add(_step())
            .add_role("x3", role="id")
            .fit(_polars())
        )
        assert fitted.schema is not None
        assert fitted.schema.transform_required["x3"] is False
        out = fitted.transform(_polars().drop("x3", "y"))
        assert "x2" in out.columns

    def test_a_step_that_read_the_column_keeps_it_required(self) -> None:
        fitted = (
            Recipe(data=_polars(), target="y", features=["x2"])
            .update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x3"]))
            .add_role("x3", role="id")
            .fit(_polars())
        )
        assert fitted.schema is not None
        assert fitted.schema.transform_required["x3"] is True


class TestBranchingAndReuse:
    def test_operations_do_not_leak_between_branches(self) -> None:
        base = Recipe(data=_polars(), target="y").add(_step())
        one = base.add_role("x1", role="g1")
        two = base.add_role("x1", role="g2")
        assert base.fit(_polars()).roles.of("x1") == {"feature"}
        assert one.fit(_polars()).roles.of("x1") == {"feature", "g1"}
        assert two.fit(_polars()).roles.of("x1") == {"feature", "g2"}

    def test_fitting_twice_gives_equal_independent_roles(self) -> None:
        rec = Recipe(data=_polars(), target="y").add(_step()).add_role("x1", role="g")
        first, second = rec.fit(_polars()), rec.fit(_polars())
        assert first.roles == second.roles
        assert first.roles is not second.roles

    def test_sklearn_clone_keeps_operations(self) -> None:
        rec = (
            Recipe(data=_polars())
            .add_role("x1", role="g")
            .add(StepNormalize(s.has_role("g")))
        )
        out = (
            clone(RecipeTransformer(rec))
            .fit(_pandas().drop(columns="y"))
            .transform(_pandas().drop(columns="y"))
        )
        assert out["x1"].round(6).tolist() != _pandas()["x1"].tolist()
        assert out["x2"].tolist() == _pandas()["x2"].tolist()


class TestProvenanceInOperations:
    def test_late_operation_selects_columns_by_source(self) -> None:
        from mlfeatures import StepDummy

        data = pd.DataFrame(
            {"c": ["u", "v", "u", "v"], "a": [1.0, 2, 3, 4], "y": [0, 1, 0, 1]}
        )
        fitted = (
            Recipe(data=data, target="y")
            .add(StepDummy(["c"]))
            .add_role(s.has_source("dummy"), role="grp")
            .fit(data)
        )
        assert "grp" in fitted.roles.of("c_v")
        assert "grp" not in fitted.roles.of("a")

    def test_leading_operation_cannot_select_a_later_step(
        self, snapshot: SnapshotAssertion
    ) -> None:
        from mlfeatures import StepDummy

        data = pd.DataFrame({"c": ["u", "v"], "y": [0, 1]})
        rec = (
            Recipe(data=data, target="y")
            .add_role(s.has_source("dummy"), role="grp")
            .add(StepDummy(["c"]))
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(data)
        assert str(excinfo.value) == snapshot


class TestRecipeTransformerTargets:
    def test_categorical_target_keeps_its_dtype(self) -> None:
        X = _pandas().drop(columns="y")
        y = pd.Series(["p", "q", "p", "q"], dtype="category")
        rt = RecipeTransformer(Recipe(target="y").add(_step())).fit(X, y)
        assert rt.fitted_.schema is not None
        assert str(rt.fitted_.schema.dtypes["y"]) == "Categorical"

    def test_nullable_integer_target_keeps_its_dtype(self) -> None:
        X = _pandas().drop(columns="y")
        y = pd.Series([1, 0, 1, None], dtype="Int64")
        rt = RecipeTransformer(Recipe(target="y").add(_step())).fit(X, y)
        assert rt.fitted_.schema is not None
        assert str(rt.fitted_.schema.dtypes["y"]) == "Int64"

    def test_target_attaches_by_position_whatever_the_index(self) -> None:
        X = _pandas().drop(columns="y")
        y = pd.Series([0, 1, 0, 1], index=[10, 11, 12, 13])
        rt = RecipeTransformer(Recipe(target="y").add(_step())).fit(X, y)
        assert rt.fitted_.roles.columns_with_role("target") == ["y"]

    def test_selector_that_removes_the_target_has_a_clear_error(
        self, snapshot: SnapshotAssertion
    ) -> None:
        X = _pandas().drop(columns="y")
        rec = (
            Recipe(target="y")
            .remove_role(s.starts_with("y"), role="target")
            .add(_step())
        )
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(rec).fit(X, pd.Series([0, 1, 0, 1]))
        assert str(excinfo.value) == snapshot

    def test_plain_name_that_removes_the_target_has_the_same_error_type(self) -> None:
        X = _pandas().drop(columns="y")
        rec = Recipe(target="y").remove_role("y", role="target").add(_step())
        with pytest.raises(ValueError, match="no 'target' role"):
            RecipeTransformer(rec).fit(X, pd.Series([0, 1, 0, 1]))


# ---------------------------------------------------------------------------
# A seeded random sweep against the model.
# ---------------------------------------------------------------------------

_ROLES = ["target", "case_weights", "group", "feature", "id"]


def _random_ops(rng: random.Random) -> list[tuple[str, str, str, str | None]]:
    ops: list[tuple[str, str, str, str | None]] = []
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["add", "remove", "update"])
        column = rng.choice(COLUMNS)
        role = rng.choice(_ROLES)
        old = (
            rng.choice([None, "feature", "group", "target"])
            if kind == "update"
            else None
        )
        ops.append((kind, column, role, old))
    return ops


@pytest.mark.parametrize("seed", range(150))
def test_random_operations_match_the_model_at_any_position(seed: int) -> None:
    rng = random.Random(seed)
    ops = _random_ops(rng)
    steps_before = rng.choice([0, 1, 2])
    expected = _model(ops)
    rec = _build(ops, steps_before=steps_before, data=_polars())
    if expected is None:
        with pytest.raises(ValueError):
            rec.fit(_polars())
        return
    fitted = rec.fit(_polars())
    assert _as_dict(fitted.roles) == expected, (seed, ops, steps_before)
    # Fitted recipes are applied to the training frame without error, and the
    # same operations written first give the same roles.
    fitted.transform(_polars())
    first = _build(ops, steps_before=0, data=_polars()).fit(_polars())
    assert _as_dict(first.roles) == expected, (seed, ops)
