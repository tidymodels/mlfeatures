"""End-to-end lifecycle tests for StepDropZeroVariance: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDropZeroVariance +
narwhals frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, SchemaInfo, StepDropZeroVariance


def _data() -> dict[str, list[float]]:
    # ``const`` is zero-variance (single unique value); ``a`` and ``b`` vary.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_drops_constant_column(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropZeroVariance(["a", "const", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_polars_drops_constant_column(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepDropZeroVariance(["a", "const", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_no_drop_when_everything_varies(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropZeroVariance(["a", "b"])])
        out = recipe.fit(df).transform(df)
        # a/b both vary, so nothing is dropped; const passes through untouched.
        assert list(out.columns) == ["a", "const", "b"]

    def test_string_constant_column_dropped(self) -> None:
        # Zero variance is type-agnostic — a constant string column is dropped.
        df = pd.DataFrame({"s": ["x", "x", "x"], "t": ["a", "b", "c"]})
        recipe = Recipe(steps=[StepDropZeroVariance(["s", "t"])])
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["t"]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropZeroVariance(["a", "const", "b"])]).fit(df)
        first = fitted.transform(df)
        second = fitted.transform(df)
        pd.testing.assert_frame_equal(first, second)

    def test_extra_columns_outside_selection_are_preserved(self) -> None:
        df = pd.DataFrame(_data())
        # Only consider a/const; b is not part of the step but must pass through.
        recipe = Recipe(steps=[StepDropZeroVariance(["a", "const"])])
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropZeroVariance(["a", "const"])]).fit(df)
        assert all(step.trained for step in fitted.steps)


class TestStepDropZeroVarianceGuards:
    def test_transform_before_fit_raises(self) -> None:
        step = StepDropZeroVariance(["a", "const"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_accepts_any_dtype(self) -> None:
        accepted = StepDropZeroVariance(["a", "const"]).accepts_dtypes()
        # ANY_DTYPE accepts everything — numeric and non-numeric alike.
        assert nw.Float64 in accepted
        assert nw.String in accepted

    def test_fit_returns_self(self) -> None:
        step = StepDropZeroVariance(["a", "const"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_default_id(self) -> None:
        assert StepDropZeroVariance(["a"]).id == "zv"
