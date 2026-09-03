"""End-to-end lifecycle tests for StepHoliday: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepHoliday + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. See ``tests/test_step_date_features_e2e.py`` for why the
pandas fixture needs a pyarrow-backed ``date32[pyarrow]`` column.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import polars as pl
import pytest
from dateutil.rrule import SA, SU, WEEKLY, rrule
from syrupy.assertion import SnapshotAssertion

from shoez import Recipe, StepHoliday, StepNormalize


def _dates() -> list[date]:
    return [date(2023, 12, 24), date(2023, 12, 25), date(2023, 1, 1)]


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
    def test_pandas_produces_holiday_indicator_columns(self) -> None:
        df = _pandas_data()
        out = (
            Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])])
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["d_christmas_day", "n"]
        assert out["d_christmas_day"].tolist() == [0, 1, 0]

    def test_polars_produces_holiday_indicator_columns(self) -> None:
        df = pl.DataFrame(_polars_data())
        out = (
            Recipe()
            .add(StepHoliday(["d"], holidays=["Christmas Day", "New Year's Day"]))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["d_christmas_day", "d_new_years_day", "n"]

    def test_unselected_columns_pass_through(self) -> None:
        df = _pandas_data()
        out = (
            Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])])
            .fit(df)
            .transform(df)
        )
        assert out["n"].tolist() == [1.0, 2.0, 3.0]

    def test_keep_original_cols_retains_source(self) -> None:
        df = _pandas_data()
        out = (
            Recipe()
            .add(
                StepHoliday(["d"], holidays=["Christmas Day"], keep_original_cols=True)
            )
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["d", "d_christmas_day", "n"]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = _pandas_data()
        fitted = Recipe(
            steps=[
                StepHoliday(["d"], holidays=["Christmas Day"], keep_original_cols=True)
            ]
        ).fit(df)
        assert "feature" in fitted.roles.of("d")
        assert "feature" in fitted.roles.of("d_christmas_day")

    def test_source_loses_roles_when_not_kept(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])]).fit(df)
        assert fitted.roles.of("d") == set()

    def test_retained_source_is_selectable_downstream(self) -> None:
        df = pd.DataFrame(
            {
                "d": pd.array(
                    [date(2018, 1, 1), date(2019, 3, 1), date(2020, 6, 1)],
                    dtype="date32[pyarrow]",
                )
            }
        )
        out = (
            Recipe()
            .add(
                StepHoliday(["d"], holidays=["New Year's Day"], keep_original_cols=True)
            )
            .add(StepNormalize(["d_new_years_day"]))
            .fit(df)
            .transform(df)
        )
        assert out["d_new_years_day"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = _pandas_data()
        fitted = Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])]).fit(
            _pandas_data()
        )
        assert all(step.trained for step in fitted.steps)

    def test_holidays_and_rules_combine_on_the_same_column(self) -> None:
        df = pl.DataFrame(_polars_data())
        weekend = rrule(WEEKLY, byweekday=(SA, SU), dtstart=date(2023, 1, 1))
        out = (
            Recipe()
            .add(
                StepHoliday(
                    ["d"], holidays=["Christmas Day"], rules={"weekend": weekend}
                )
            )
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["d_christmas_day", "d_weekend", "n"]
        # _dates(): Dec 24 (Sun), Dec 25 (Mon), Jan 1 (Sun)
        assert out["d_christmas_day"].to_list() == [0, 1, 0]
        assert out["d_weekend"].to_list() == [1, 0, 1]

    def test_to_json_raises_for_rules(self, snapshot: SnapshotAssertion) -> None:
        weekend = rrule(WEEKLY, byweekday=(SA, SU), dtstart=date(2023, 1, 1))
        fitted = Recipe(
            steps=[StepHoliday(["d"], holidays=[], rules={"weekend": weekend})]
        ).fit(_pandas_data())
        with pytest.raises(TypeError) as excinfo:
            fitted.to_json()
        assert str(excinfo.value) == snapshot

    def test_new_data_beyond_training_years_is_still_classified(self) -> None:
        train = _pandas_data()
        fitted = Recipe(steps=[StepHoliday(["d"], holidays=["Christmas Day"])]).fit(
            train
        )
        new_data = pd.DataFrame(
            {"d": pd.array([date(2030, 12, 25)], dtype="date32[pyarrow]"), "n": [9.0]}
        )
        out = fitted.transform(new_data)
        assert out["d_christmas_day"].tolist() == [1]
