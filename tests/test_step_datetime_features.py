"""Tests for :mod:`shoez.step_datetime_features`.

Datetime columns are natively representable on both pandas (``datetime64``)
and polars (``Datetime``) from a plain list of ``datetime.datetime`` objects,
so these tests build inputs directly as pandas frames -- no pyarrow needed
(compare ``tests/test_step_date_features.py``, which does need it).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest

from shoez import Mode, SchemaInfo
from shoez.step_datetime_features import (
    DatetimeFeaturesFit,
    StepDatetimeFeatures,
    apply_datetime_features,
    fit_datetime_features,
)


def _data() -> dict[str, list[Any]]:
    return {
        "dt": [
            datetime(2020, 1, 1, 8, 30, 15),
            datetime(2020, 6, 15, 13, 45, 30),
            datetime(2020, 12, 31, 23, 59, 59),
        ],
        "n": [1.0, 2.0, 3.0],
    }


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_datetime_features / apply_datetime_features
# ---------------------------------------------------------------------------


class TestFitDatetimeFeatures:
    def test_records_columns_and_features(self) -> None:
        frame = _df({"dt": _data()["dt"]})
        result = fit_datetime_features(frame, ("year", "month"))
        assert result == DatetimeFeaturesFit(columns=["dt"], features=("year", "month"))


class TestApplyDatetimeFeatures:
    def test_produces_expected_values(self) -> None:
        frame = _df({"dt": _data()["dt"]})
        fit_result = fit_datetime_features(
            frame, ("year", "month", "day", "hour", "minute", "second", "dow", "doy")
        )
        out = apply_datetime_features(frame, fit_result).to_native()
        assert out["dt_year"].tolist() == [2020, 2020, 2020]
        assert out["dt_month"].tolist() == [1, 6, 12]
        assert out["dt_day"].tolist() == [1, 15, 31]
        assert out["dt_hour"].tolist() == [8, 13, 23]
        assert out["dt_minute"].tolist() == [30, 45, 59]
        assert out["dt_second"].tolist() == [15, 30, 59]
        assert out["dt_dow"].tolist() == [3, 1, 4]
        assert out["dt_doy"].tolist() == [1, 167, 366]

    def test_drops_source_column_by_default(self) -> None:
        frame = _df({"dt": _data()["dt"], "n": _data()["n"]})
        fit_result = fit_datetime_features(_df({"dt": _data()["dt"]}), ("year",))
        out = apply_datetime_features(frame, fit_result).to_native()
        assert "dt" not in out.columns
        assert list(out.columns) == ["dt_year", "n"]

    def test_keep_original_cols_retains_source(self) -> None:
        frame = _df({"dt": _data()["dt"], "n": _data()["n"]})
        fit_result = fit_datetime_features(_df({"dt": _data()["dt"]}), ("year",))
        out = apply_datetime_features(
            frame, fit_result, keep_original_cols=True
        ).to_native()
        assert list(out.columns) == ["dt", "dt_year", "n"]

    def test_output_dtype_is_int32(self) -> None:
        frame = _df({"dt": _data()["dt"]})
        fit_result = fit_datetime_features(frame, ("year", "hour"))
        out = apply_datetime_features(frame, fit_result)
        assert str(out.schema["dt_year"]) == "Int32"
        assert str(out.schema["dt_hour"]) == "Int32"


# ---------------------------------------------------------------------------
# StepDatetimeFeatures — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepDatetimeFeaturesUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepDatetimeFeatures(["dt"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepDatetimeFeatures(["dt"]).id == "datetime_features"

    def test_explicit_id(self) -> None:
        assert StepDatetimeFeatures(["dt"], id="my_dt").id == "my_dt"

    def test_default_features(self) -> None:
        assert StepDatetimeFeatures(["dt"]).features == (
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dow",
            "doy",
        )

    def test_features_are_reordered_canonically(self) -> None:
        step = StepDatetimeFeatures(["dt"], features=["doy", "dow", "hour", "year"])
        assert step.features == ("year", "hour", "dow", "doy")

    def test_duplicate_features_are_deduped(self) -> None:
        step = StepDatetimeFeatures(["dt"], features=["year", "year", "hour"])
        assert step.features == ("year", "hour")

    def test_default_role_is_feature(self) -> None:
        assert StepDatetimeFeatures(["dt"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepDatetimeFeatures(["dt"]).keep_original_cols is False

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError):
            StepDatetimeFeatures(["dt"], features=["not_a_feature"])

    def test_fit_returns_self(self) -> None:
        step = StepDatetimeFeatures(["dt"])
        fitted = step.fit(_df(_data()), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepDatetimeFeatures(["dt"])
        with pytest.raises(RuntimeError):
            step.transform(_df(_data()))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDatetimeFeatures(["dt"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDatetimeFeatures(["dt"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDatetimeFeatures(["dt"]).output_parents()

    def test_accepts_datetime_dtype_only(self) -> None:
        accepted = StepDatetimeFeatures(["dt"]).accepts_dtypes()
        assert nw.Datetime in accepted
        assert nw.Date not in accepted
        assert nw.Float64 not in accepted

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepDatetimeFeatures(["dt"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_in().columns == ["dt"]

    def test_schema_out_lists_calendar_part_columns(self) -> None:
        step = StepDatetimeFeatures(["dt"], features=["year", "month"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_out().columns == ["dt_year", "dt_month"]
        assert step.schema_out().dtypes == {"dt_year": "Int32", "dt_month": "Int32"}

    def test_update_roles_with_no_output_role(self) -> None:
        from shoez.types import Roles

        step = StepDatetimeFeatures(["dt"], features=["year"], role=None)
        step.fit(_df(_data()), SchemaInfo())
        roles = Roles()
        roles.add("dt", "feature")
        step.update_roles(roles)
        assert roles.of("dt_year") == set()
        assert roles.of("dt") == set()

    def test_output_parents_maps_each_column_to_its_source(self) -> None:
        step = StepDatetimeFeatures(["dt"], features=["year", "month"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.output_parents() == {"dt_year": ["dt"], "dt_month": ["dt"]}

    def test_not_sparse(self) -> None:
        assert StepDatetimeFeatures(["dt"]).produces_sparse is False
