"""End-to-end lifecycle tests for StepDatetimeFeatures: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDatetimeFeatures +
narwhals frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, StepDatetimeFeatures, StepNormalize


def _data() -> dict[str, list[Any]]:
    return {
        "dt": [
            datetime(2020, 1, 1, 8, 30, 15),
            datetime(2020, 6, 15, 13, 45, 30),
            datetime(2020, 12, 31, 23, 59, 59),
        ],
        "n": [1.0, 2.0, 3.0],
    }


class TestEndToEnd:
    def test_pandas_produces_calendar_part_columns(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepDatetimeFeatures(["dt"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == [
            "dt_year",
            "dt_month",
            "dt_day",
            "dt_hour",
            "dt_minute",
            "dt_second",
            "dt_dow",
            "dt_doy",
            "n",
        ]

    def test_polars_produces_calendar_part_columns(self) -> None:
        df = pl.DataFrame(_data())
        out = (
            Recipe()
            .add(StepDatetimeFeatures(["dt"], features=["year", "month"]))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["dt_year", "dt_month", "n"]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepDatetimeFeatures(["dt"])]).fit(df).transform(df)
        assert out["n"].tolist() == [1.0, 2.0, 3.0]

    def test_keep_original_cols_retains_source(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(
                StepDatetimeFeatures(["dt"], features=["year"], keep_original_cols=True)
            )
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["dt", "dt_year", "n"]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepDatetimeFeatures(["dt"], features=["year"], keep_original_cols=True)
            ]
        ).fit(df)
        assert "feature" in fitted.roles.of("dt")
        assert "feature" in fitted.roles.of("dt_year")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDatetimeFeatures(["dt"])]).fit(df)
        assert fitted.roles.of("dt") == set()

    def test_retained_source_is_selectable_downstream(self) -> None:
        df = pd.DataFrame(
            {
                "dt": [
                    datetime(2018, 1, 1),
                    datetime(2019, 1, 1),
                    datetime(2020, 1, 1),
                ]
            }
        )
        out = (
            Recipe()
            .add(
                StepDatetimeFeatures(["dt"], features=["year"], keep_original_cols=True)
            )
            .add(StepNormalize(["dt_year"]))
            .fit(df)
            .transform(df)
        )
        assert out["dt_year"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDatetimeFeatures(["dt"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepDatetimeFeatures(["dt"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)
