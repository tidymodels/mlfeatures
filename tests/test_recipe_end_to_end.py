"""End-to-end lifecycle tests: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDropCorr + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, SchemaInfo, StepDropCorr


def _data() -> dict[str, list[float]]:
    # ``b`` is a perfect linear function of ``a`` (r == 1.0); ``c`` is unrelated.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_drops_correlated_column(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # Exactly one of the perfectly-correlated pair is removed; c survives.
        assert "c" in out.columns
        assert len(out.columns) == 2
        assert ("a" in out.columns) ^ ("b" in out.columns)

    def test_polars_drops_correlated_column(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepDropCorr(["a", "b", "c"], threshold=0.9))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert "c" in out.columns
        assert len(out.columns) == 2
        assert ("a" in out.columns) ^ ("b" in out.columns)

    def test_no_drop_when_threshold_above_all_pairs(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropCorr(["a", "c"], threshold=0.99)])
        out = recipe.fit(df).transform(df)
        # a/c are weakly correlated, so nothing is dropped; b passes through.
        assert list(out.columns) == ["a", "b", "c"]

    def test_add_is_pure(self) -> None:
        base = Recipe()
        extended = base.add(StepDropCorr(["a", "b"]))
        assert base.steps == []
        assert len(extended.steps) == 1
        assert base is not extended

    def test_add_is_variadic(self) -> None:
        recipe = Recipe().add(StepDropCorr(["a", "b"]), StepDropCorr(["a", "c"]))
        assert len(recipe.steps) == 2

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropCorr(["a", "b", "c"])]).fit(df)
        first = fitted.transform(df)
        second = fitted.transform(df)
        pd.testing.assert_frame_equal(first, second)

    def test_fit_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropCorr(["a", "b", "c"])])
        out1 = recipe.fit(df).transform(df)
        out2 = recipe.fit(df).transform(df)
        pd.testing.assert_frame_equal(out1, out2)

    def test_extra_columns_outside_selection_are_preserved(self) -> None:
        df = pd.DataFrame(_data())
        # Only consider a/b; c is not part of the step but must pass through.
        recipe = Recipe(steps=[StepDropCorr(["a", "b"], threshold=0.9)])
        out = recipe.fit(df).transform(df)
        assert "c" in out.columns
        assert len(out.columns) == 2

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropCorr(["a", "b"])]).fit(df)
        assert all(step.trained for step in fitted.steps)


class TestStepDropCorrGuards:
    def test_transform_before_fit_raises(self) -> None:
        step = StepDropCorr(["a", "b"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_accepts_narwhals_numeric_dtypes(self) -> None:
        accepted = StepDropCorr(["a", "b"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        # A non-numeric narwhals dtype is rejected.
        assert nw.String not in accepted

    def test_fit_returns_self(self) -> None:
        step = StepDropCorr(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step
