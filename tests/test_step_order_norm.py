"""Tests for fit_order_norm and apply_order_norm."""

from __future__ import annotations

import math
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_order_norm import apply_order_norm, fit_order_norm


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mapping_of_fits(self) -> None:
        fits = fit_order_norm(_df(("a", [1.0, 2.0, 3.0, 4.0])))
        assert list(fits) == ["a"]

    def test_one_entry_per_column(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        fits = fit_order_norm(df)
        assert list(fits) == ["a", "b"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_columns_returns_empty(self) -> None:
        fits = fit_order_norm(nw.from_native(pd.DataFrame(), eager_only=True))
        assert fits == {}

    def test_zero_rows_is_degenerate(self) -> None:
        df = nw.from_native(
            pd.DataFrame({"a": pd.Series([], dtype="float64")}), eager_only=True
        )
        with pytest.warns(UserWarning):
            fits = fit_order_norm(df)
        assert fits == {}

    def test_single_row_is_degenerate(self) -> None:
        with pytest.warns(UserWarning):
            fits = fit_order_norm(_df(("a", [42.0])))
        assert fits == {}


# ---------------------------------------------------------------------------
# Core estimates (verified against R bestNormalize::orderNorm)
# ---------------------------------------------------------------------------


class TestCoreEstimates:
    def test_matches_bestnormalize_reference(self) -> None:
        # Reference computed with R bestNormalize::orderNorm(x, warn = FALSE)$x.t
        # on x <- c(1, 2, 4, 8, 16, 3, 5, 9, 2, 7).
        x = [1.0, 2.0, 4.0, 8.0, 16.0, 3.0, 5.0, 9.0, 2.0, 7.0]
        expected = [
            -1.644854,
            -0.841621,
            -0.125661,
            0.674490,
            1.644854,
            -0.385320,
            0.125661,
            1.036433,
            -0.841621,
            0.385320,
        ]
        fits = fit_order_norm(_df(("a", x)))
        out = apply_order_norm(_df(("a", x)), fits)
        assert out["a"].to_list() == pytest.approx(expected, abs=1e-5)

    def test_output_is_monotonic_in_input(self) -> None:
        x = [3.0, 7.0, 11.0, 2.0, 9.0, 0.5]
        fits = fit_order_norm(_df(("a", x)))
        out = apply_order_norm(_df(("a", x)), fits)
        order = np.argsort(x)
        assert list(np.argsort(out["a"].to_list())) == list(order)

    def test_column_order_preserved(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0, 4.0]), ("a", [10.0, 20.0, 30.0, 40.0]))
        fits = fit_order_norm(df)
        assert list(fits) == ["b", "a"]

    def test_integer_column_accepted(self) -> None:
        fits = fit_order_norm(_df(("a", [1, 2, 3, 4])))
        assert list(fits) == ["a"]

    def test_ties_get_average_rank(self) -> None:
        # Two tied values in the middle must map to the same z-score.
        x = [1.0, 2.0, 2.0, 3.0]
        fits = fit_order_norm(_df(("a", x)))
        out = apply_order_norm(_df(("a", x)), fits)
        values = out["a"].to_list()
        assert values[1] == pytest.approx(values[2])

    def test_missing_values_ignored_at_fit(self) -> None:
        x = [1.0, 2.0, 3.0, float("nan")]
        fits = fit_order_norm(_df(("a", x)))
        assert fits["a"].x.size == 3


# ---------------------------------------------------------------------------
# Degenerate (fewer than 2 distinct values) columns are dropped with a warning
# ---------------------------------------------------------------------------


class TestDegenerateColumns:
    def test_constant_column_is_omitted(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("x", [1.0, 2.0, 3.0, 4.0]), ("c", [5.0, 5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning) as record:
            fits = fit_order_norm(df)
        assert str(record[0].message) == snapshot
        assert list(fits) == ["x"]

    def test_warning_names_the_columns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning) as record:
            fits = fit_order_norm(df)
        assert str(record[0].message) == snapshot
        assert fits == {}

    def test_no_warning_when_no_degenerate_columns(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fits = fit_order_norm(_df(("x", [1.0, 2.0, 3.0, 4.0])))
        assert list(fits) == ["x"]


# ---------------------------------------------------------------------------
# apply_order_norm
# ---------------------------------------------------------------------------


class TestApplyOrderNorm:
    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0, 4.0]), ("a", [10.0, 20.0, 30.0, 40.0]))
        out = apply_order_norm(df, fit_order_norm(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_in_range_value_maps_to_zero_at_median(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0, 4.0, 5.0]))
        out = apply_order_norm(df, fit_order_norm(df))
        assert out["a"].to_list()[2] == pytest.approx(0.0, abs=1e-9)

    def test_missing_values_pass_through_as_missing(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0, float("nan")]))
        out = apply_order_norm(df, fit_order_norm(df))
        assert math.isnan(out["a"].to_list()[-1])

    def test_degenerate_column_passed_through_untouched(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0, 4.0]), ("c", [5.0, 5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning):
            fitted = fit_order_norm(df)
        out = apply_order_norm(df, fitted)
        assert out["c"].to_list() == [5.0, 5.0, 5.0, 5.0]

    def test_empty_fits_is_passthrough(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("y", [4.0, 5.0, 6.0]))
        out = apply_order_norm(df, {})
        assert out["x"].to_list() == [1.0, 2.0, 3.0]
        assert out["y"].to_list() == [4.0, 5.0, 6.0]

    def test_integer_input_yields_float_output(self) -> None:
        df = _df(("a", [1, 2, 3, 4]))
        assert df["a"].dtype.is_integer()
        out = apply_order_norm(df, fit_order_norm(df))
        assert out["a"].dtype.is_float()

    def test_extrapolates_below_range_using_boundary_slope(self) -> None:
        train = _df(("a", [1.0, 2.0, 3.0, 4.0, 5.0]))
        fits = fit_order_norm(train)
        new = _df(("a", [0.0]))
        out = apply_order_norm(new, fits)
        # Slope of the lowest segment (a=1 -> a=2) extended one unit below a=1.
        fit = fits["a"]
        lo_slope = (fit.z[1] - fit.z[0]) / (fit.x[1] - fit.x[0])
        expected = fit.z[0] + (0.0 - fit.x[0]) * lo_slope
        assert out["a"].to_list()[0] == pytest.approx(expected)

    def test_extrapolates_above_range_using_boundary_slope(self) -> None:
        train = _df(("a", [1.0, 2.0, 3.0, 4.0, 5.0]))
        fits = fit_order_norm(train)
        new = _df(("a", [100.0]))
        out = apply_order_norm(new, fits)
        fit = fits["a"]
        hi_slope = (fit.z[-1] - fit.z[-2]) / (fit.x[-1] - fit.x[-2])
        expected = fit.z[-1] + (100.0 - fit.x[-1]) * hi_slope
        assert out["a"].to_list()[0] == pytest.approx(expected)

    def test_out_of_range_value_is_further_from_zero_than_boundary(self) -> None:
        train = _df(("a", [1.0, 2.0, 3.0, 4.0, 5.0]))
        fits = fit_order_norm(train)
        out = apply_order_norm(_df(("a", [-10.0, 50.0])), fits)
        low, high = out["a"].to_list()
        assert low < fits["a"].z[0]
        assert high > fits["a"].z[-1]
