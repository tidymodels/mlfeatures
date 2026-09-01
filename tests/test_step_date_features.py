"""Tests for :mod:`shoez.step_date_features`.

Unlike Datetime, pandas cannot represent narwhals' ``Date`` dtype from a plain
list of ``datetime.date`` objects without pyarrow (it infers ``object``
dtype instead) -- see ``tests/test_selectors.py``'s ``_typed_frame_with_roles``
for the same limitation. These tests build inputs as polars frames, which
represent ``Date`` natively; ``StepDateFeatures`` itself is backend-agnostic
(see `test_step_date_features_infrastructure.py`, which exercises the pandas
side via pyarrow and `shoez.testing`'s `dtype_casts` hook).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import narwhals.stable.v2 as nw
import polars as pl
import pytest

from shoez import Mode, SchemaInfo
from shoez.step_date_features import (
    DateFeaturesFit,
    StepDateFeatures,
    apply_date_features,
    fit_date_features,
)


def _data() -> dict[str, list[Any]]:
    return {
        "d": [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)],
        "n": [1.0, 2.0, 3.0],
    }


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pl.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_date_features / apply_date_features
# ---------------------------------------------------------------------------


class TestFitDateFeatures:
    def test_records_columns_and_features(self) -> None:
        frame = _df({"d": _data()["d"]})
        result = fit_date_features(frame, ("year", "month"))
        assert result == DateFeaturesFit(columns=["d"], features=("year", "month"))


class TestApplyDateFeatures:
    def test_produces_expected_values(self) -> None:
        frame = _df({"d": _data()["d"]})
        fit_result = fit_date_features(frame, ("year", "month", "day", "dow", "doy"))
        out = apply_date_features(frame, fit_result).to_native()
        assert out["d_year"].to_list() == [2020, 2020, 2020]
        assert out["d_month"].to_list() == [1, 6, 12]
        assert out["d_day"].to_list() == [1, 15, 31]
        assert out["d_dow"].to_list() == [3, 1, 4]
        assert out["d_doy"].to_list() == [1, 167, 366]

    def test_drops_source_column_by_default(self) -> None:
        frame = _df({"d": _data()["d"], "n": _data()["n"]})
        fit_result = fit_date_features(_df({"d": _data()["d"]}), ("year",))
        out = apply_date_features(frame, fit_result).to_native()
        assert "d" not in out.columns
        assert list(out.columns) == ["d_year", "n"]

    def test_keep_original_cols_retains_source(self) -> None:
        frame = _df({"d": _data()["d"], "n": _data()["n"]})
        fit_result = fit_date_features(_df({"d": _data()["d"]}), ("year",))
        out = apply_date_features(
            frame, fit_result, keep_original_cols=True
        ).to_native()
        assert list(out.columns) == ["d", "d_year", "n"]

    def test_output_dtype_is_int32(self) -> None:
        frame = _df({"d": _data()["d"]})
        fit_result = fit_date_features(frame, ("year", "doy"))
        out = apply_date_features(frame, fit_result)
        assert str(out.schema["d_year"]) == "Int32"
        assert str(out.schema["d_doy"]) == "Int32"


# ---------------------------------------------------------------------------
# StepDateFeatures — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepDateFeaturesUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepDateFeatures(["d"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepDateFeatures(["d"]).id == "date_features"

    def test_explicit_id(self) -> None:
        assert StepDateFeatures(["d"], id="my_date").id == "my_date"

    def test_default_features(self) -> None:
        assert StepDateFeatures(["d"]).features == (
            "year",
            "month",
            "day",
            "dow",
            "doy",
        )

    def test_default_role_is_feature(self) -> None:
        assert StepDateFeatures(["d"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepDateFeatures(["d"]).keep_original_cols is False

    def test_invalid_feature_raises(self) -> None:
        with pytest.raises(ValueError):
            StepDateFeatures(["d"], features=["not_a_feature"])

    def test_fit_returns_self(self) -> None:
        step = StepDateFeatures(["d"])
        fitted = step.fit(_df(_data()), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepDateFeatures(["d"])
        with pytest.raises(RuntimeError):
            step.transform(_df(_data()))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDateFeatures(["d"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDateFeatures(["d"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepDateFeatures(["d"]).output_parents()

    def test_accepts_date_dtype_only(self) -> None:
        accepted = StepDateFeatures(["d"]).accepts_dtypes()
        assert nw.Date in accepted
        assert nw.Datetime not in accepted
        assert nw.Float64 not in accepted

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepDateFeatures(["d"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_in().columns == ["d"]

    def test_schema_out_lists_calendar_part_columns(self) -> None:
        step = StepDateFeatures(["d"], features=["year", "month"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_out().columns == ["d_year", "d_month"]
        assert step.schema_out().dtypes == {"d_year": "Int32", "d_month": "Int32"}

    def test_update_roles_with_no_output_role(self) -> None:
        from shoez.types import Roles

        step = StepDateFeatures(["d"], features=["year"], role=None)
        step.fit(_df(_data()), SchemaInfo())
        roles = Roles()
        roles.add("d", "feature")
        step.update_roles(roles)
        assert roles.of("d_year") == set()
        assert roles.of("d") == set()

    def test_output_parents_maps_each_column_to_its_source(self) -> None:
        step = StepDateFeatures(["d"], features=["year", "month"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.output_parents() == {"d_year": ["d"], "d_month": ["d"]}

    def test_not_sparse(self) -> None:
        assert StepDateFeatures(["d"]).produces_sparse is False

    def test_required_pkgs(self) -> None:
        assert StepDateFeatures(["d"]).required_pkgs() == ["pyarrow"]
