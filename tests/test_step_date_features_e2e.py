"""End-to-end lifecycle tests for StepDateFeatures: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDateFeatures + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Because pandas cannot represent narwhals' ``Date`` dtype
from a plain list of ``datetime.date`` objects (it infers ``object`` dtype
instead), the pandas fixture is built via a pyarrow-backed
``pd.array(..., dtype="date32[pyarrow]")`` -- see
``tests/test_step_date_features_infrastructure.py`` for the same pattern
applied through `mlfeatures.testing`'s `dtype_casts` hook.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import polars as pl
import pytest

from mlfeatures import Recipe, StepDateFeatures, StepNormalize


def _dates() -> list[date]:
    return [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)]


def _pandas_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "d": pd.array(_dates(), dtype="date32[pyarrow]"),
            "n": [1.0, 2.0, 3.0],
        }
    )


def _polars_data() -> dict[str, list[Any]]:
    return {"d": _dates(), "n": [1.0, 2.0, 3.0]}


class TestEndToEnd:
    def test_pandas_produces_calendar_part_columns(self) -> None:
        df = _pandas_data()
        out = Recipe(steps=[StepDateFeatures(["d"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == [
            "d_year",
            "d_month",
            "d_day",
            "d_dow",
            "d_doy",
            "n",
        ]

    def test_polars_produces_calendar_part_columns(self) -> None:
        df = pl.DataFrame(_polars_data())
        out = (
            Recipe()
            .add(StepDateFeatures(["d"], features=["year", "month"]))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["d_year", "d_month", "n"]

    def test_unselected_columns_pass_through(self) -> None:
        df = _pandas_data()
        out = Recipe(steps=[StepDateFeatures(["d"])]).fit(df).transform(df)
        assert out["n"].tolist() == [1.0, 2.0, 3.0]

    def test_keep_original_cols_retains_source(self) -> None:
        df = _pandas_data()
        out = (
            Recipe()
            .add(StepDateFeatures(["d"], features=["year"], keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["d", "d_year", "n"]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = _pandas_data()
        fitted = Recipe(
            steps=[StepDateFeatures(["d"], features=["year"], keep_original_cols=True)]
        ).fit(df)
        assert "feature" in fitted.roles.of("d")
        assert "feature" in fitted.roles.of("d_year")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepDateFeatures(["d"])]).fit(df)
        assert fitted.roles.of("d") == set()

    def test_retained_source_is_selectable_downstream(self) -> None:
        df = pd.DataFrame(
            {
                "d": pd.array(
                    [date(2018, 1, 1), date(2019, 1, 1), date(2020, 1, 1)],
                    dtype="date32[pyarrow]",
                )
            }
        )
        out = (
            Recipe()
            .add(StepDateFeatures(["d"], features=["year"], keep_original_cols=True))
            .add(StepNormalize(["d_year"]))
            .fit(df)
            .transform(df)
        )
        assert out["d_year"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepDateFeatures(["d"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepDateFeatures(["d"])]).fit(_pandas_data())
        assert all(step.trained for step in fitted.steps)
