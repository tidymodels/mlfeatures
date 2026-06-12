"""Native shoez selector DSL (issue #20).

Covers the native by-name / by-type / by-role surface, set-op composition and
column ordering, cross-family coercion + ``Selectors.wrap``, the provenance
selectors ``has_source`` / ``has_ancestor`` resolving against the lineage DAG,
zero-match silence, and the documented limit that a polars/narwhals selector
cannot lead a cross-family expression.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest

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
        roles.add(col, "predictor")
    roles.add("y", "outcome")
    return frame, roles


def _resolve(selector: object, frame: Frame, roles: Roles) -> list[str]:
    return _frame.select(frame, selector, roles).columns  # type: ignore[arg-type]


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


class TestByRole:
    def test_predictors_and_outcomes(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.predictors(), frame, roles) == [
            "x1",
            "x2",
            "count",
            "label",
        ]
        assert _resolve(Selectors.outcomes(), frame, roles) == ["y"]

    def test_has_role_and_compound(self) -> None:
        frame, roles = _frame_with_roles()
        assert _resolve(Selectors.has_role("outcome"), frame, roles) == ["y"]
        assert _resolve(Selectors.predictors_numeric(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]


class TestSetOps:
    def test_and_or_not_difference(self) -> None:
        frame, roles = _frame_with_roles()
        # intersection
        assert _resolve(Selectors.numeric() & Selectors.predictors(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]
        # union
        assert _resolve(Selectors.outcomes() | Selectors.string(), frame, roles) == [
            "label",
            "y",
        ]
        # complement
        assert _resolve(~Selectors.outcomes(), frame, roles) == [
            "x1",
            "x2",
            "count",
            "label",
        ]
        # difference
        assert _resolve(Selectors.numeric() - Selectors.outcomes(), frame, roles) == [
            "x1",
            "x2",
            "count",
        ]

    def test_ordering_follows_frame_not_operands(self) -> None:
        frame, roles = _frame_with_roles()
        # Even though "y" is listed first in the union operand, frame order wins.
        result = _resolve(Selectors.col("y") | Selectors.col("x1"), frame, roles)
        assert result == ["x1", "y"]


class TestCrossFamily:
    def test_shoez_leads_coerces_polars(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.predictors() & cs.starts_with("x"), frame, roles)
        assert result == ["x1", "x2"]

    def test_shoez_leads_coerces_list(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(Selectors.numeric() & ["x1", "label"], frame, roles)
        assert result == ["x1"]

    def test_wrap_lets_polars_lead(self) -> None:
        frame, roles = _frame_with_roles()
        result = _resolve(
            Selectors.wrap(cs.starts_with("x")) & Selectors.predictors(), frame, roles
        )
        assert result == ["x1", "x2"]

    def test_polars_leading_is_unsupported(self) -> None:
        # Documented limit: a polars selector on the left cannot see the shoez
        # operand and polars raises rather than composing.
        with pytest.raises(TypeError):
            _ = cs.numeric() & Selectors.predictors()  # type: ignore[operator]


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

    def test_zero_match_is_silent_noop(self) -> None:
        df = _dummy_data()
        # No StepOneHot in the pipeline, so this matches nothing: identity.
        out = (
            Recipe()
            .add(StepNormalize(Selectors.has_source(step_type=StepDummy)))
            .fit(df)
            .transform(df)
        )
        pd.testing.assert_frame_equal(out, df)

    def test_resolution_without_dag_yields_nothing(self) -> None:
        # Outside a recipe fit there is no DAG; provenance resolves to empty.
        frame, roles = _frame_with_roles()
        sel = Selectors.has_source(step_type=StepDummy)
        assert sel.resolve(frame, roles) == []

    def test_requires_exactly_one_argument(self) -> None:
        with pytest.raises(ValueError):
            Selectors.has_source()
        with pytest.raises(ValueError):
            Selectors.has_source("a", step_type=StepDummy)


class TestPolarsEscapeHatch:
    def test_raw_polars_selector_still_accepted(self) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = Recipe(steps=[StepNormalize(cs.numeric())]).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert abs(nw.from_native(out)["a"].mean()) < 1e-9
