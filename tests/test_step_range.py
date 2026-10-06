"""Tests for fit_range, apply_range, and the StepRange guards."""

from __future__ import annotations

import builtins
import math
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import StepRange
from mlfeatures.step_range import apply_range, fit_range


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

    def test_matches_the_closed_form_on_new_data(self) -> None:
        # The reference formula, spelled out: rescale into [min, max] using the
        # training bounds, then clamp. Mirrors the check in R recipes'
        # test-range.R, where held-out rows fall outside the fitted bounds.
        train = _df(("x", [2.0, 4.0, 6.0, 10.0]))  # bounds (2, 10)
        new = _df(("x", [1.0, 4.0, 6.0, 11.0]))
        new_min, new_max = -12.0, 1.0
        out = apply_range(new, fit_range(train), min=new_min, max=new_max)

        lo, hi = 2.0, 10.0
        expected = [
            builtins.min(
                builtins.max(
                    (v - lo) / (hi - lo) * (new_max - new_min) + new_min, new_min
                ),
                new_max,
            )
            for v in [1.0, 4.0, 6.0, 11.0]
        ]
        assert out["x"].to_list() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_nulls_are_excluded_from_the_bounds(self) -> None:
        lo, hi = fit_range(_df(("a", [3.0, None, 1.0, 7.0])))["a"]
        assert (lo, hi) == (1.0, 7.0)

    def test_null_rows_stay_missing_after_rescaling(self) -> None:
        df = _df(("x", [1.0, None, 3.0]))
        out = apply_range(df, fit_range(df))
        values = out["x"].to_list()
        assert values[0] == pytest.approx(0.0)
        assert math.isnan(values[1])
        assert values[2] == pytest.approx(1.0)

    def test_null_rows_stay_missing_without_clipping(self) -> None:
        # Clipping must not turn a missing value into a bound.
        df = _df(("x", [1.0, None, 3.0]))
        out = apply_range(df, fit_range(df), clipping=False)
        assert math.isnan(out["x"].to_list()[1])

    def test_all_null_column_yields_nan_bounds(self) -> None:
        # No non-null values to compare, so the column is not "constant" — it
        # is kept with NaN bounds, the same signal as a zero-row fit.
        bounds = fit_range(_df(("a", [None, None, None])))
        assert list(bounds) == ["a"]
        lo, hi = bounds["a"]
        assert math.isnan(lo)
        assert math.isnan(hi)

    def test_constant_apart_from_nulls_is_still_constant(self) -> None:
        with pytest.warns(UserWarning):
            assert fit_range(_df(("c", [5.0, 5.0, None]))) == {}

    def test_nan_bounds_propagate_to_every_row(self) -> None:
        out = apply_range(_df(("x", [1.0, 2.0])), {"x": (float("nan"), float("nan"))})
        assert all(math.isnan(v) for v in out["x"].to_list())


# ---------------------------------------------------------------------------
# Infinite values
# ---------------------------------------------------------------------------


class TestInfiniteValues:
    def test_positive_infinity_becomes_the_upper_bound(self) -> None:
        with pytest.warns(UserWarning):
            lo, hi = fit_range(_df(("a", [2.0, 3.0, float("inf")])))["a"]
        assert lo == 2.0
        assert hi == float("inf")

    def test_negative_infinity_becomes_the_lower_bound(self) -> None:
        with pytest.warns(UserWarning):
            lo, hi = fit_range(_df(("a", [2.0, 3.0, float("-inf")])))["a"]
        assert lo == float("-inf")
        assert hi == 3.0

    def test_warning_names_the_columns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("i", [2.0, 3.0, float("inf")]))
        with pytest.warns(UserWarning) as record:
            bounds = fit_range(df)
        assert str(record[0].message) == snapshot
        assert list(bounds) == ["x", "i"]  # kept, unlike a constant column

    def test_constant_and_infinite_columns_warn_separately(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = _df(("c", [5.0, 5.0, 5.0]), ("i", [2.0, 3.0, float("inf")]))
        with pytest.warns(UserWarning) as record:
            fit_range(df)
        assert [str(w.message) for w in record] == snapshot

    def test_no_warning_when_all_bounds_are_finite(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert list(fit_range(_df(("x", [1.0, 2.0, 3.0])))) == ["x"]

    def test_infinite_bound_degrades_the_rescaling(self) -> None:
        # An infinite range divides every finite deviation to 0, and the
        # infinite value itself becomes NaN (inf - inf) — which is what the
        # fit-time warning is for.
        df = _df(("a", [2.0, 3.0, float("inf")]))
        with pytest.warns(UserWarning):
            bounds = fit_range(df)
        values = apply_range(df, bounds)["a"].to_list()
        assert values[0] == 0.0
        assert values[1] == 0.0
        assert math.isnan(values[2])
