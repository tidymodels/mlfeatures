"""End-to-end lifecycle tests for StepIndicateNa: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepIndicateNa + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import math

import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, StepIndicateNa, StepNormalize


def _pandas_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [1.0, None, 3.0],
            "n": [1.0, 2.0, 3.0],
        }
    )


def _polars_data() -> dict[str, list[float | None]]:
    return {"x": [1.0, None, 3.0], "n": [1.0, 2.0, 3.0]}


class TestEndToEnd:
    def test_pandas_produces_indicator_columns(self) -> None:
        df = _pandas_data()
        out = Recipe(steps=[StepIndicateNa(["x"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["x_na", "n"]
        assert out["x_na"].tolist() == [0, 1, 0]

    def test_polars_produces_indicator_columns(self) -> None:
        df = pl.DataFrame(_polars_data())
        out = Recipe().add(StepIndicateNa(["x"])).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["x_na", "n"]
        assert out["x_na"].to_list() == [0, 1, 0]

    def test_unselected_columns_pass_through(self) -> None:
        df = _pandas_data()
        out = Recipe(steps=[StepIndicateNa(["x"])]).fit(df).transform(df)
        assert out["n"].tolist() == [1.0, 2.0, 3.0]

    def test_keep_original_cols_retains_source(self) -> None:
        df = _pandas_data()
        out = (
            Recipe()
            .add(StepIndicateNa(["x"], keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["x", "x_na", "n"]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepIndicateNa(["x"], keep_original_cols=True)]).fit(df)
        assert "feature" in fitted.roles.of("x")
        assert "feature" in fitted.roles.of("x_na")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepIndicateNa(["x"])]).fit(df)
        assert fitted.roles.of("x") == set()

    def test_new_data_missing_values_are_flagged_even_if_training_had_none(
        self,
    ) -> None:
        train = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        new = pd.DataFrame({"x": [1.0, math.nan, None]})
        fitted = Recipe(steps=[StepIndicateNa(["x"])]).fit(train)
        out = fitted.transform(new)
        assert out["x_na"].tolist() == [0, 1, 1]

    def test_retained_source_is_selectable_downstream(self) -> None:
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        out = (
            Recipe()
            .add(StepIndicateNa(["x"], keep_original_cols=True))
            .add(StepNormalize(["x_na"]))
            .fit(df)
            .transform(df)
        )
        assert out["x_na"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepIndicateNa(["x"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepIndicateNa(["x"])]).fit(_pandas_data())
        assert all(step.trained for step in fitted.steps)
