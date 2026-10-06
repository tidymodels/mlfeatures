"""Native shoez selector DSL (issue #20).

Covers the native by-name / by-type / by-role surface, set-op composition and
column ordering, cross-family coercion + ``Selectors.wrap``, the provenance
selectors ``has_source`` / ``has_ancestor`` resolving against the lineage DAG,
zero-match silence, and the documented limit that a polars/narwhals selector
cannot lead a cross-family expression.
"""

from __future__ import annotations

import copy

import narwhals.selectors as ncs
import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest
from narwhals.exceptions import ColumnNotFoundError
from syrupy.assertion import SnapshotAssertion

from shoez import Recipe, Selectors, StepDummy, StepNormalize, _frame
from shoez.types import Frame, Roles


def _frame_with_roles() -> tuple[Frame, Roles]:
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0],
            "x2": [4.0, 5.0, 6.0],
            "count": [1, 2, 3],
            "label": ["a", "b", "c"],
            "y": [0.0, 1.0, 0.0],
        }
    )
    frame = _frame.from_native(df)
    roles = Roles()
    for col in ("x1", "x2", "count", "label"):
        roles.add(col, "feature")
    roles.add("y", "target")
    return frame, roles


def _typed_frame_with_roles() -> tuple[Frame, Roles]:
    """A frame spanning every dtype family the type selectors target.

    pandas cannot represent a narwhals ``Date`` without pyarrow, so this fixture
    is polars-backed; the selector layer is backend-agnostic. Column order is
    fixed so ``resolve`` results can be asserted exactly. ``y`` (float) and
    ``grp`` (categorical) are targets so the ``*_features`` compounds have a
    same-typed non-feature column to exclude.
    """
    df = pl.DataFrame(
        {
            "fl": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
            "i": pl.Series([1, 2, 3], dtype=pl.Int64),
            "s": ["x", "y", "z"],
            "cat": pl.Series(["a", "b", "c"], dtype=pl.Categorical),
            "enum": pl.Series(["a", "b", "c"], dtype=pl.Enum(["a", "b", "c"])),
            "flag": [True, False, True],
            "d": pl.Series(["2020-01-01", "2020-01-02", "2020-01-03"]).str.to_date(),
            "dt": pl.Series(
                ["2020-01-01", "2020-01-02", "2020-01-03"]
            ).str.to_datetime(),
            "y": pl.Series([0.0, 1.0, 0.0], dtype=pl.Float64),
            "grp": pl.Series(["p", "q", "p"], dtype=pl.Categorical),
        }
    )
    frame = _frame.from_native(df)
    roles = Roles()
    for col in ("fl", "i", "s", "cat", "enum", "flag", "d", "dt"):
        roles.add(col, "feature")
    roles.add("y", "target")
    roles.add("grp", "target")
    return frame, roles


def _resolve(selector: object, frame: Frame, roles: Roles) -> list[str]:
    return _frame.select(frame, selector, roles).columns  # type: ignore[arg-type]


class TestTypedFixture:
    def test_everything_spans_all_columns(self) -> None:
        frame, roles = _typed_frame_with_roles()
        assert _resolve(Selectors.everything(), frame, roles) == _frame.columns(frame)


class TestByName:
    def test_col(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.col("x1", "y"), frame, roles) == ["x1", "y"]

    def test_col_missing_is_silent(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.col("nope"), frame, roles) == []

    def test_starts_ends_contains(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.starts_with("x"), frame, roles) == ["x1", "x2"]
        assert _resolve(Selectors.ends_with("1"), frame, roles) == ["x1"]
        assert _resolve(Selectors.contains("ab"), frame, roles) == ["label"]

    def test_matches_regex(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.matches(r"^x\d+$"), frame, roles) == ["x1", "x2"]

    def test_num_range(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.num_range("x", range(1, 6)), frame, roles) == [
            "x1",
            "x2",
        ]

    def test_everything(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.everything(), frame, roles) == _frame.columns(frame)

    def test_name_predicates_are_case_sensitive(self) -> None:
        frame, roles = _frame_with_roles()
        # Columns are lower-case x1/x2/label; upper-case queries match nothing.
        assert _resolve(Selectors.starts_with("X"), frame, roles) == []
        assert _resolve(Selectors.contains("AB"), frame, roles) == []
        assert _resolve(Selectors.matches(r"^X"), frame, roles) == []

    def test_num_range_empty_and_no_match(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.num_range("x", range(0)), frame, roles) == []
        assert _resolve(Selectors.num_range("x", range(7, 9)), frame, roles) == []

    def test_everything_under_complement_and_difference(self) -> None:
        frame, roles = _frame_with_roles()
        # Complement of everything() is empty; everything() minus a column drops it.
        assert _resolve(~Selectors.everything(), frame, roles) == []
        assert _resolve(Selectors.everything() - Selectors.col("y"), frame, roles) == [
            "x1",
            "x2",
            "count",
            "label",
        ]


class TestByType:
    def test_numeric_integer_float(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.numeric(), frame, roles) == ["x1", "x2", "count", "y"]
        assert _resolve(Selectors.integer(), frame, roles) == ["count"]
        assert _resolve(Selectors.float(), frame, roles) == ["x1", "x2", "y"]

    def test_string_and_nominal(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.string(), frame, roles) == ["label"]
        assert _resolve(Selectors.nominal(), frame, roles) == ["label"]


class TestByTypeFullDtypeRange:
    """Every type selector against a frame spanning all dtype families."""

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            (Selectors.numeric(), ["fl", "i", "y"]),
            (Selectors.integer(), ["i"]),
            (Selectors.float(), ["fl", "y"]),
            (Selectors.double(), ["fl", "y"]),  # alias of float()
            (Selectors.string(), ["s"]),
            (Selectors.categorical(), ["cat", "enum", "grp"]),
            (Selectors.ordered(), ["enum"]),
            (Selectors.unordered(), ["cat", "grp"]),
            (Selectors.nominal(), ["s", "cat", "enum", "grp"]),
            (Selectors.boolean(), ["flag"]),
            (Selectors.date(), ["d"]),
            (Selectors.datetime(), ["dt"]),
        ],
    )
    def test_type_selector(self, selector: object, expected: list[str]) -> None:
        frame, roles = _typed_frame_with_roles()
        assert _resolve(selector, frame, roles) == expected

    def test_nominal_distinguishes_from_string(self) -> None:
        # nominal() must include categorical/enum that string() excludes.
        frame, roles = _typed_frame_with_roles()
        assert _resolve(Selectors.string(), frame, roles) == ["s"]
        assert _resolve(Selectors.nominal(), frame, roles) == [
            "s",
            "cat",
            "enum",
            "grp",
        ]


class TestByRole:
    def test_features_and_targets(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.features(), frame, roles) == [
            "x1",
            "x2",
            "count",
            "label",
        ]
        assert _resolve(Selectors.targets(), frame, roles) == ["y"]

    def test_has_role_and_compound(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.has_role("target"), frame, roles) == ["y"]
        assert _resolve(Selectors.numeric_features(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]

    def test_has_role_custom_role(self) -> None:
        # has_role resolves an arbitrary, non-built-in role.
        frame, _ = _frame_with_roles()
        roles = Roles()
        roles.add("x1", "id")
        roles.add("count", "id")
        roles.add("x2", "feature")
        assert _resolve(Selectors.has_role("id"), frame, roles) == ["x1", "count"]


class TestCompoundFeatures:
    """The ``*_features`` compounds = type selector intersected with features.

    ``y`` (float) and ``grp`` (categorical) are targets in the fixture, so
    ``numeric_features`` / ``double_features`` exclude ``y`` and
    ``categorical_features`` / ``nominal_features`` exclude ``grp`` — proving
    the role filter actually bites, not just the type filter.
    """

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            (Selectors.numeric_features(), ["fl", "i"]),
            (Selectors.integer_features(), ["i"]),
            (Selectors.double_features(), ["fl"]),
            (Selectors.string_features(), ["s"]),
            (Selectors.categorical_features(), ["cat", "enum"]),
            (Selectors.nominal_features(), ["s", "cat", "enum"]),
            (Selectors.boolean_features(), ["flag"]),
            (Selectors.date_features(), ["d"]),
            (Selectors.datetime_features(), ["dt"]),
        ],
    )
    def test_compound_feature(self, selector: object, expected: list[str]) -> None:
        frame, roles = _typed_frame_with_roles()
        assert _resolve(selector, frame, roles) == expected


class TestMissingNameStrictness:
    """`col` tolerates absent names; a bare list operand is strict.

    This mirrors recipes' ``any_of`` (tolerant) vs ``all_of`` (strict) split:
    ``Selectors.col`` silently drops names that aren't present, but a plain list
    operand goes through ``Frame.select`` and raises on a missing column.
    """

    def test_col_drops_missing_names(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.col("x1", "missing"), frame, roles) == ["x1"]

    def test_list_operand_raises_on_missing(self) -> None:
        frame, roles = _frame_with_roles()
        with pytest.raises(ColumnNotFoundError):
            _resolve(Selectors.numeric() & ["x1", "missing"], frame, roles)


class TestSetOps:
    def test_and_or_not_difference(self) -> None:
        frame, roles = _frame_with_roles()
        # intersection
        assert _resolve(Selectors.numeric() & Selectors.features(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]
        # union
        assert _resolve(Selectors.targets() | Selectors.string(), frame, roles) == [
            "label",
            "y",
        ]
        # complement
        assert _resolve(~Selectors.targets(), frame, roles) == [
            "x1",
            "x2",
            "count",
            "label",
        ]
        # difference
        assert _resolve(Selectors.numeric() - Selectors.targets(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]

    def test_ordering_follows_frame_not_operands(self) -> None:
        frame, roles = _frame_with_roles()
        # Even though "y" is listed first in the union operand, frame order wins.
        result = _resolve(Selectors.col("y") | Selectors.col("x1"), frame, roles)
        assert result == ["x1", "y"]

    def test_added_then_removed_dedup(self) -> None:
        frame, roles = _frame_with_roles()
        # Add x2 in the union, then subtract it: it must not survive.
        result = _resolve(
            (Selectors.col("x1") | Selectors.col("x2")) - Selectors.col("x2"),
            frame,
            roles,
        )
        assert result == ["x1"]

    def test_overlapping_union_dedups(self) -> None:
        frame, roles = _frame_with_roles()
        # numeric and float overlap on x1/x2; each column appears once.
        result = _resolve(Selectors.numeric() | Selectors.float(), frame, roles)
        assert result == ["x1", "x2", "count", "y"]

    def test_complement_of_composite(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(~(Selectors.targets() | Selectors.string()), frame, roles)
        assert result == ["x1", "x2", "count"]

    def test_three_operand_chain(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(
            Selectors.numeric() & Selectors.features() & Selectors.starts_with("x"),
            frame,
            roles,
        )
        assert result == ["x1", "x2"]


class TestCrossFamily:
    def test_shoez_leads_coerces_polars(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.features() & cs.starts_with("x"), frame, roles)
        assert result == ["x1", "x2"]

    def test_shoez_leads_coerces_list(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.numeric() & ["x1", "label"], frame, roles)
        assert result == ["x1"]

    def test_shoez_leads_coerces_narwhals(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.features() & ncs.numeric(), frame, roles)
        assert result == ["x1", "x2", "count"]

    def test_cross_family_union(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.targets() | cs.starts_with("x"), frame, roles)
        assert result == ["x1", "x2", "y"]

    def test_cross_family_difference(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.numeric() - cs.starts_with("x"), frame, roles)
        assert result == ["count", "y"]

    def test_wrap_lets_polars_lead(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(
            Selectors.wrap(cs.starts_with("x")) & Selectors.features(), frame, roles
        )
        assert result == ["x1", "x2"]

    def test_wrap_lets_narwhals_lead(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(
            Selectors.wrap(ncs.numeric()) & Selectors.col("x1"), frame, roles
        )
        assert result == ["x1"]

    def test_wrap_accepts_list(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(
            Selectors.wrap(["x1", "label"]) | Selectors.col("x2"), frame, roles
        )
        assert result == ["x1", "x2", "label"]

    def test_wrap_accepts_bare_name(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.wrap("x1"), frame, roles) == ["x1"]

    def test_polars_leading_is_unsupported(self) -> None:
        # Documented limit: a polars selector on the left cannot see the shoez
        # operand and polars raises rather than composing.
        with pytest.raises(TypeError):
            _ = cs.numeric() & Selectors.features()  # type: ignore[operator]


def _dummy_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "color": ["red", "green", "blue", "red"],
            "size": [1.0, 2.0, 3.0, 4.0],
        }
    )


class TestProvenance:
    def test_has_source_direct(self) -> None:
        df = _dummy_data()
        fitted = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_source(step_type=StepDummy), id="capture"))
            .fit(df)
        )
        captured = fitted.steps[-1].schema_in().columns
        assert sorted(captured) == ["color_green", "color_red"]

    def test_has_source_by_id(self) -> None:
        df = _dummy_data()
        fitted = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_source("dummy"), id="capture"))
            .fit(df)
        )
        assert sorted(fitted.steps[-1].schema_in().columns) == [
            "color_green",
            "color_red",
        ]

    def test_has_ancestor_traces_through_intervening_step(self) -> None:
        df = _dummy_data()
        # Normalize the dummies first; has_source would then miss them (their
        # immediate producer changed) but has_ancestor still traces back.
        fitted = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_source(step_type=StepDummy), id="norm"))
            .add(
                StepNormalize(Selectors.has_ancestor(step_type=StepDummy), id="capture")
            )
            .fit(df)
        )
        assert sorted(fitted.steps[-1].schema_in().columns) == [
            "color_green",
            "color_red",
        ]

    def test_unknown_step_type_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _dummy_data()
        # No StepDummy in the pipeline: a mistake knowable from the recipe
        # alone, so it fails loudly instead of becoming the identity transform.
        recipe = Recipe().add(StepNormalize(Selectors.has_source(step_type=StepDummy)))
        with pytest.raises(ValueError) as excinfo:
            recipe.fit(df)
        assert str(excinfo.value) == snapshot

    def test_resolution_without_dag_yields_nothing(self) -> None:
        # Outside a recipe fit there is no DAG; provenance resolves to empty.
        frame, roles = _frame_with_roles()
        sel = Selectors.has_source(step_type=StepDummy)
        assert sel.resolve(frame, roles) == []

    def test_has_ancestor_by_id(self) -> None:
        df = _dummy_data()
        fitted = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_source(step_type=StepDummy), id="norm"))
            .add(StepNormalize(Selectors.has_ancestor("dummy"), id="capture"))
            .fit(df)
        )
        assert sorted(fitted.steps[-1].schema_in().columns) == [
            "color_green",
            "color_red",
        ]

    def test_has_source_unknown_id_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _dummy_data()
        # An id that matches no step in the pipeline is a typo, not a selection
        # that happens to be empty, so it is reported rather than skipped. The
        # message lists the ids that were available.
        recipe = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_source("nonexistent")))
        )
        with pytest.raises(ValueError) as excinfo:
            recipe.fit(df)
        assert str(excinfo.value) == snapshot

    def test_has_ancestor_unknown_id_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _dummy_data()
        # The check lives in the shared _SourceSelector, so has_ancestor too.
        recipe = (
            Recipe()
            .add(StepDummy(["color"], id="dummy"))
            .add(StepNormalize(Selectors.has_ancestor("nonexistent")))
        )
        with pytest.raises(ValueError) as excinfo:
            recipe.fit(df)
        assert str(excinfo.value) == snapshot

    def test_has_source_original_is_always_valid(self) -> None:
        df = _dummy_data()
        # "original" is not a step id but is a legitimate source: the input
        # columns that survive to this point.
        fitted = (
            Recipe()
            .add(
                StepNormalize(
                    Selectors.has_source("original") & Selectors.numeric_features(),
                    id="capture",
                )
            )
            .fit(df)
        )
        assert fitted.steps[-1].schema_in().columns == ["size"]

    def test_id_of_a_later_step_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _dummy_data()
        # Only steps fitted before this one are visible; a later step has
        # produced no columns yet, so the id is reported as unavailable even
        # though the recipe does contain it.
        recipe = (
            Recipe()
            .add(StepNormalize(Selectors.has_source("dummy")))
            .add(StepDummy(["color"], id="dummy"))
        )
        with pytest.raises(ValueError) as excinfo:
            recipe.fit(df)
        assert str(excinfo.value) == snapshot

    def test_requires_exactly_one_argument(self) -> None:
        # Validation lives in the shared _SourceSelector, so it applies to both
        # has_source and has_ancestor.
        for factory in (Selectors.has_source, Selectors.has_ancestor):
            with pytest.raises(ValueError):
                factory()
            with pytest.raises(ValueError):
                factory("a", step_type=StepDummy)


class TestPolarsEscapeHatch:
    def test_raw_polars_selector_still_accepted(self) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = Recipe(steps=[StepNormalize(cs.numeric())]).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert abs(nw.from_native(out)["a"].mean()) < 1e-9

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [
            (pl.col("x1"), ["x1"]),
            (pl.col("x1", "count"), ["x1", "count"]),
            (pl.col(pl.Float64), ["x1", "x2", "y"]),
            (pl.all().exclude("label", "y"), ["x1", "x2", "count"]),
            (cs.numeric() - cs.by_name("y"), ["x1", "x2", "count"]),
        ],
        ids=["col", "col-names", "col-dtype", "all-exclude", "selector"],
    )
    def test_column_selection_expression_resolves(
        self, selector: pl.Expr, expected: list[str]
    ) -> None:
        # The fixture frame is pandas-backed: resolution is against the schema,
        # so a polars expression works whatever the frame's backend.
        frame, roles = _frame_with_roles()
        assert _resolve(selector, frame, roles) == expected

    @pytest.mark.parametrize(
        "selector", [pl.col("x1"), cs.by_name("x1")], ids=["col", "selector"]
    )
    def test_resolves_after_deepcopy(self, selector: pl.Expr) -> None:
        # `Recipe.fit` deep-copies each step, and so its selector.
        frame, roles = _frame_with_roles()
        assert _resolve(copy.deepcopy(selector), frame, roles) == ["x1"]

    @pytest.mark.parametrize(
        "selector",
        [pl.col("x1") + 1, pl.col("x1").alias("renamed")],
        ids=["computed", "alias"],
    )
    def test_non_selection_expression_raises(
        self, selector: pl.Expr, snapshot: SnapshotAssertion
    ) -> None:
        frame, roles = _frame_with_roles()
        with pytest.raises(TypeError) as excinfo:
            _resolve(selector, frame, roles)
        assert str(excinfo.value) == snapshot

    def test_shoez_leads_coerces_column_expression(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.features() & pl.col("x1", "y"), frame, roles)
        assert result == ["x1"]

    def test_column_expression_in_recipe(self) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = Recipe().add(StepNormalize(pl.col("a"))).fit(df).transform(df)
        assert abs(nw.from_native(out)["a"].mean()) < 1e-9
        assert out["b"].to_list() == [3.0, 4.0]


class TestInRecipe:
    """Native selectors driving a real step through Recipe.fit().transform().

    The other suites resolve selectors directly; these prove the same selectors
    pick the right columns end to end, including role plumbing through the recipe.
    """

    def _df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "x1": [1.0, 2.0, 3.0],
                "x2": [10.0, 20.0, 30.0],
                "tag": ["a", "b", "c"],
                "y": [0.0, 10.0, 20.0],
            }
        )

    def test_role_selector_only_touches_features(self) -> None:
        df = self._df()
        recipe = Recipe(features=["x1", "x2"], target="y").add(
            StepNormalize(Selectors.numeric_features())
        )
        out = recipe.fit(df).transform(df)
        # features normalized (mean ~ 0); target left untouched.
        assert abs(out["x1"].mean()) < 1e-9
        assert abs(out["x2"].mean()) < 1e-9
        assert out["y"].tolist() == [0.0, 10.0, 20.0]
        assert out["tag"].tolist() == ["a", "b", "c"]

    def test_name_selector_drives_step(self) -> None:
        df = self._df()
        out = (
            Recipe()
            .add(StepNormalize(Selectors.starts_with("x")))
            .fit(df)
            .transform(df)
        )
        assert abs(out["x1"].mean()) < 1e-9
        assert abs(out["x2"].mean()) < 1e-9
        assert out["y"].tolist() == [0.0, 10.0, 20.0]

    def test_set_op_selector_drives_step(self) -> None:
        df = self._df()
        recipe = Recipe(features=["x1", "x2"], target="y").add(
            StepNormalize(Selectors.numeric() & Selectors.features())
        )
        out = recipe.fit(df).transform(df)
        assert abs(out["x1"].mean()) < 1e-9
        assert out["y"].tolist() == [0.0, 10.0, 20.0]

    def test_empty_native_selection_is_passthrough(self) -> None:
        df = self._df()
        out = (
            Recipe()
            .add(StepNormalize(Selectors.starts_with("zzz")))
            .fit(df)
            .transform(df)
        )
        pd.testing.assert_frame_equal(out, df)
