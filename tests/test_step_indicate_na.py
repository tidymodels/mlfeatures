"""Tests for :mod:`mlfeatures.step_indicate_na`."""

from __future__ import annotations

import math
from typing import Any

import narwhals.stable.v2 as nw
import polars as pl
import pytest

from mlfeatures import Mode, SchemaInfo
from mlfeatures.step_indicate_na import (
    IndicateNaFit,
    StepIndicateNa,
    apply_indicate_na,
    fit_indicate_na,
)
from mlfeatures.types import Roles


def _data() -> dict[str, list[Any]]:
    return {
        "x": [1.0, None, math.nan],
        "g": ["a", None, "b"],
        "n": [1.0, 2.0, 3.0],
    }


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pl.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_indicate_na / apply_indicate_na
# ---------------------------------------------------------------------------


class TestFitIndicateNA:
    def test_records_columns_and_dtypes(self) -> None:
        frame = _df({"x": _data()["x"]})
        result = fit_indicate_na(frame)
        assert result == IndicateNaFit(columns=["x"], dtypes={"x": nw.Float64()})


class TestApplyIndicateNA:
    def test_flags_null_as_missing(self) -> None:
        frame = _df({"g": _data()["g"]})
        fit_result = fit_indicate_na(frame)
        out = apply_indicate_na(frame, fit_result).to_native()
        assert out["g_na"].to_list() == [0, 1, 0]

    def test_flags_null_and_nan_as_missing_for_floats(self) -> None:
        frame = _df({"x": _data()["x"]})
        fit_result = fit_indicate_na(frame)
        out = apply_indicate_na(frame, fit_result).to_native()
        assert out["x_na"].to_list() == [0, 1, 1]

    def test_drops_source_column_by_default(self) -> None:
        frame = _df(_data())
        fit_result = fit_indicate_na(_df({"x": _data()["x"]}))
        out = apply_indicate_na(frame, fit_result).to_native()
        assert "x" not in out.columns
        assert list(out.columns) == ["x_na", "g", "n"]

    def test_keep_original_cols_retains_source(self) -> None:
        frame = _df(_data())
        fit_result = fit_indicate_na(_df({"x": _data()["x"]}))
        out = apply_indicate_na(frame, fit_result, keep_original_cols=True).to_native()
        assert list(out.columns) == ["x", "x_na", "g", "n"]

    def test_output_dtype_is_int32(self) -> None:
        frame = _df({"x": _data()["x"]})
        fit_result = fit_indicate_na(frame)
        out = apply_indicate_na(frame, fit_result)
        assert str(out.schema["x_na"]) == "Int32"

    def test_no_missing_values_produces_all_zeros(self) -> None:
        frame = _df({"n": _data()["n"]})
        fit_result = fit_indicate_na(frame)
        out = apply_indicate_na(frame, fit_result).to_native()
        assert out["n_na"].to_list() == [0, 0, 0]


# ---------------------------------------------------------------------------
# StepIndicateNa: unit-level invariants
# ---------------------------------------------------------------------------


class TestStepIndicateNaUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepIndicateNa(["x"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepIndicateNa(["x"]).id == "indicate_na"

    def test_explicit_id(self) -> None:
        assert StepIndicateNa(["x"], id="my_na").id == "my_na"

    def test_default_role_is_feature(self) -> None:
        assert StepIndicateNa(["x"]).output_role == "feature"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepIndicateNa(["x"]).keep_original_cols is False

    def test_fit_returns_self(self) -> None:
        step = StepIndicateNa(["x"])
        fitted = step.fit(_df(_data()), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepIndicateNa(["x"])
        with pytest.raises(RuntimeError):
            step.transform(_df(_data()))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepIndicateNa(["x"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepIndicateNa(["x"]).schema_out()

    def test_output_parents_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepIndicateNa(["x"]).output_parents()

    def test_accepts_any_dtype(self) -> None:
        from mlfeatures.types import ANY_DTYPE

        assert StepIndicateNa(["x"]).accepts_dtypes() is ANY_DTYPE

    def test_schema_in_lists_selected_columns(self) -> None:
        step = StepIndicateNa(["x"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_in().columns == ["x"]

    def test_schema_out_lists_indicator_columns(self) -> None:
        step = StepIndicateNa(["x", "g"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.schema_out().columns == ["x_na", "g_na"]
        assert step.schema_out().dtypes == {"x_na": "Int32", "g_na": "Int32"}

    def test_update_roles_with_no_output_role(self) -> None:
        step = StepIndicateNa(["x"], role=None)
        step.fit(_df(_data()), SchemaInfo())
        roles = Roles()
        roles.add("x", "feature")
        step.update_roles(roles)
        assert roles.of("x_na") == set()
        assert roles.of("x") == set()

    def test_output_parents_maps_each_column_to_its_source(self) -> None:
        step = StepIndicateNa(["x", "g"])
        step.fit(_df(_data()), SchemaInfo())
        assert step.output_parents() == {"x_na": ["x"], "g_na": ["g"]}

    def test_not_sparse(self) -> None:
        assert StepIndicateNa(["x"]).produces_sparse is False

    def test_required_pkgs_is_empty(self) -> None:
        assert StepIndicateNa(["x"]).required_pkgs() == []
