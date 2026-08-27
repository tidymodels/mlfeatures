"""Tests for fit_range, apply_range, and the StepRange guards."""

from __future__ import annotations

import math
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import StepRange
from shoez.step_range import apply_range, fit_range


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mapping_of_lo_hi_pairs(self) -> None:
        bounds = fit_range(_df(("a", [1.0, 2.0, 3.0])))
        assert list(bounds) == ["a"]
        lo, hi = bounds["a"]
        assert isinstance(lo, float)
        assert isinstance(hi, float)

    def test_one_entry_per_column(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        assert list(fit_range(df)) == ["a", "b"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_columns_returns_empty(self) -> None:
        assert fit_range(nw.from_native(pd.DataFrame(), eager_only=True)) == {}

    def test_zero_rows_yields_nan(self) -> None:
        df = nw.from_native(
            pd.DataFrame({"a": pd.Series([], dtype="float64")}), eager_only=True
        )
        bounds = fit_range(df)
        # NaN bounds are not treated as constant — the column is kept.
        assert list(bounds) == ["a"]
        lo, hi = bounds["a"]
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_min_not_less_than_max_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepRange(["a"], min=1.0, max=0.0)
        assert str(excinfo.value) == snapshot

    def test_equal_min_and_max_raises(self) -> None:
        with pytest.raises(ValueError):
            StepRange(["a"], min=1.0, max=1.0)

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepRange(["a"]).transform(_df(("a", [1.0, 2.0])))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepRange(["a"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepRange(["a"]).schema_out()


# ---------------------------------------------------------------------------
# Core estimates
# ---------------------------------------------------------------------------


class TestCoreEstimates:
    def test_known_bounds(self) -> None:
        lo, hi = fit_range(_df(("a", [3.0, 1.0, 7.0, 2.0])))["a"]
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(7.0)

    def test_column_order_preserved(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        bounds = fit_range(df)
        assert list(bounds) == ["b", "a"]

    def test_integer_column_accepted(self) -> None:
        lo, hi = fit_range(_df(("a", [1, 2, 3])))["a"]
        assert (lo, hi) == (1.0, 3.0)

    def test_single_row_is_constant(self) -> None:
        with pytest.warns(UserWarning):
            assert fit_range(_df(("a", [42.0]))) == {}


# ---------------------------------------------------------------------------
# Constant (zero-range) columns are dropped with a warning
# ---------------------------------------------------------------------------


class TestConstantColumns:
    def test_constant_column_is_omitted(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning) as record:
            bounds = fit_range(df)
        assert str(record[0].message) == snapshot
        assert list(bounds) == ["x"]  # constant "c" dropped

    def test_warning_names_the_columns(self, snapshot: SnapshotAssertion) -> None:
        with pytest.warns(UserWarning) as record:
            bounds = fit_range(_df(("c", [5.0, 5.0, 5.0])))
        assert str(record[0].message) == snapshot
        assert bounds == {}

    def test_no_warning_when_no_constant_columns(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert list(fit_range(_df(("x", [1.0, 2.0, 3.0])))) == ["x"]


# ---------------------------------------------------------------------------
# apply_range
# ---------------------------------------------------------------------------


class TestApplyRange:
    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        out = apply_range(df, fit_range(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_scales_to_unit_interval(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0, 5.0]))
        out = apply_range(df, fit_range(df))
        assert out["x"].to_list() == pytest.approx([0.0, 0.25, 0.5, 1.0])

    def test_custom_target_range(self) -> None:
        df = _df(("x", [1.0, 3.0, 5.0]))
        out = apply_range(df, fit_range(df), min=-1.0, max=1.0)
        assert out["x"].to_list() == pytest.approx([-1.0, 0.0, 1.0])

    def test_clipping_clamps_out_of_range_values(self) -> None:
        train = _df(("x", [1.0, 3.0, 5.0]))
        new = _df(("x", [0.0, 3.0, 9.0]))
        out = apply_range(new, fit_range(train))
        assert out["x"].to_list() == pytest.approx([0.0, 0.5, 1.0])

    def test_clipping_false_extrapolates(self) -> None:
        train = _df(("x", [1.0, 3.0, 5.0]))
        new = _df(("x", [0.0, 3.0, 9.0]))
        out = apply_range(new, fit_range(train), clipping=False)
        assert out["x"].to_list() == pytest.approx([-0.25, 0.5, 2.0])

    def test_clipping_respects_custom_bounds(self) -> None:
        train = _df(("x", [1.0, 5.0]))
        new = _df(("x", [-10.0, 10.0]))
        out = apply_range(new, fit_range(train), min=-1.0, max=1.0)
        assert out["x"].to_list() == pytest.approx([-1.0, 1.0])

    def test_constant_column_passed_through_untouched(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning):
            bounds = fit_range(df)
        out = apply_range(df, bounds)
        assert out["c"].to_list() == [5.0, 5.0, 5.0]
        assert out["x"].to_list() == pytest.approx([0.0, 0.5, 1.0])

    def test_integer_input_yields_float_output(self) -> None:
        df = _df(("a", [1, 2, 3, 5]))
        assert df["a"].dtype.is_integer()  # integer in
        out = apply_range(df, fit_range(df))
        assert out["a"].dtype.is_float()  # float/double out
        assert out["a"].dtype == nw.Float64
        # values are real-valued (not truncated back to ints)
        assert out["a"].to_list() == pytest.approx([0.0, 0.25, 0.5, 1.0])

    def test_untouched_integer_column_stays_integer(self) -> None:
        df = _df(("x", [1, 2, 3, 5]), ("c", [7, 7, 7, 7]))
        with pytest.warns(UserWarning):
            bounds = fit_range(df)
        assert list(bounds) == ["x"]  # constant "c" not rescaled
        out = apply_range(df, bounds)
        assert out["c"].dtype.is_integer()  # int in -> int out (untouched)
        assert out["c"].to_list() == [7, 7, 7, 7]
        assert out["x"].dtype.is_float()  # rescaled column promoted to float

    def test_empty_bounds_is_passthrough(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("y", [4.0, 5.0, 6.0]))
        out = apply_range(df, {})
        assert out["x"].to_list() == [1.0, 2.0, 3.0]
        assert out["y"].to_list() == [4.0, 5.0, 6.0]
