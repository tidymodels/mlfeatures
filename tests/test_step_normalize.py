"""Tests for fit_normalize and apply_normalize."""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from shoez.step_normalize import apply_normalize, fit_normalize


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_columns_and_two_float_arrays(self) -> None:
        columns, means, stds = fit_normalize(_df(("a", [1.0, 2.0, 3.0])))
        assert columns == ["a"]
        assert means.dtype == np.float64
        assert stds.dtype == np.float64

    def test_arrays_parallel_to_columns(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        columns, means, stds = fit_normalize(df)
        assert columns == ["a", "b"]
        assert means.shape == (2,)
        assert stds.shape == (2,)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_columns_returns_empties(self) -> None:
        columns, means, stds = fit_normalize(
            nw.from_native(pd.DataFrame(), eager_only=True)
        )
        assert columns == []
        assert means.shape == (0,)
        assert stds.shape == (0,)
        assert means.dtype == np.float64

    def test_zero_rows_yields_nan(self) -> None:
        df = nw.from_native(
            pd.DataFrame({"a": pd.Series([], dtype="float64")}), eager_only=True
        )
        columns, means, stds = fit_normalize(df)
        # NaN std is not treated as constant — the column is kept.
        assert columns == ["a"]
        assert np.isnan(means[0])
        assert np.isnan(stds[0])

    def test_single_row_yields_nan_std(self) -> None:
        columns, means, stds = fit_normalize(_df(("a", [42.0])))
        assert columns == ["a"]
        assert means[0] == 42.0
        assert np.isnan(stds[0])


# ---------------------------------------------------------------------------
# Core estimates
# ---------------------------------------------------------------------------


class TestCoreEstimates:
    def test_known_mean_and_sample_std(self) -> None:
        # [1, 2, 3]: mean 2, sample std (ddof=1) = 1.0.
        _, means, stds = fit_normalize(_df(("a", [1.0, 2.0, 3.0])))
        assert means[0] == pytest.approx(2.0)
        assert stds[0] == pytest.approx(1.0)

    def test_column_order_preserved(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        columns, means, _ = fit_normalize(df)
        assert columns == ["b", "a"]
        assert means[0] == pytest.approx(2.0)
        assert means[1] == pytest.approx(20.0)

    def test_integer_column_accepted(self) -> None:
        _, means, stds = fit_normalize(_df(("a", [1, 2, 3])))
        assert means[0] == pytest.approx(2.0)
        assert stds[0] == pytest.approx(1.0)

    def test_normalized_column_has_unit_std(self) -> None:
        # The invariant: scaling by the returned (ddof=1) std must make the
        # normalized column's sample std exactly 1.
        col = _df(("a", [3.0, 7.0, 11.0, 2.0, 9.0]))
        _, means, stds = fit_normalize(col)
        normalized = (col["a"] - means[0]) / stds[0]
        assert normalized.std(ddof=1) == pytest.approx(1.0)
        assert normalized.mean() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero-variance (constant) columns are dropped with a warning
# ---------------------------------------------------------------------------


class TestConstantColumns:
    def test_constant_column_is_omitted(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning, match="zero standard deviation"):
            columns, means, stds = fit_normalize(df)
        assert columns == ["x"]  # constant "c" dropped
        assert means.shape == (1,)
        assert stds.shape == (1,)

    def test_warning_names_the_columns(self) -> None:
        df = _df(("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning, match=r"'c'"):
            columns, _, _ = fit_normalize(df)
        assert columns == []

    def test_no_warning_when_no_constant_columns(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            columns, _, _ = fit_normalize(_df(("x", [1.0, 2.0, 3.0])))
        assert columns == ["x"]


# ---------------------------------------------------------------------------
# apply_normalize
# ---------------------------------------------------------------------------


class TestApplyNormalize:
    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        out = apply_normalize(df, *fit_normalize(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_each_column_centered_and_unit_std(self) -> None:
        df = _df(("x", [3.0, 7.0, 11.0, 2.0, 9.0]), ("y", [-1.0, 0.0, 1.0, 5.0, 4.0]))
        out = apply_normalize(df, *fit_normalize(df))
        for col in out.columns:
            assert out[col].mean() == pytest.approx(0.0)
            assert out[col].std(ddof=1) == pytest.approx(1.0)

    def test_round_trip_with_fit(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0, 4.0]))
        columns, means, stds = fit_normalize(df)
        out = apply_normalize(df, columns, means, stds)
        expected = [(v - means[0]) / stds[0] for v in [1.0, 2.0, 3.0, 4.0]]
        assert out["x"].to_list() == pytest.approx(expected)

    def test_constant_column_passed_through_untouched(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning):
            fitted = fit_normalize(df)
        out = apply_normalize(df, *fitted)
        # "c" was dropped at fit time, so it is not normalized — left as-is.
        assert out["c"].to_list() == [5.0, 5.0, 5.0]
        assert out["x"].mean() == pytest.approx(0.0)

    def test_integer_input_yields_float_output(self) -> None:
        df = _df(("a", [1, 2, 3, 4]))
        assert df["a"].dtype.is_integer()  # integer in
        out = apply_normalize(df, *fit_normalize(df))
        assert out["a"].dtype.is_float()  # float/double out
        assert out["a"].dtype == nw.Float64
        # values are real-valued (not truncated back to ints)
        assert out["a"].mean() == pytest.approx(0.0)
        assert out["a"].std(ddof=1) == pytest.approx(1.0)

    def test_untouched_integer_column_stays_integer(self) -> None:
        # A column apply_normalize leaves alone (here a dropped constant column)
        # keeps its original integer dtype: int in -> int out.
        df = _df(("x", [1, 2, 3, 4]), ("c", [7, 7, 7, 7]))
        with pytest.warns(UserWarning):
            cols, means, stds = fit_normalize(df)
        assert cols == ["x"]  # constant "c" not normalized
        out = apply_normalize(df, cols, means, stds)
        assert out["c"].dtype.is_integer()  # int in -> int out (untouched)
        assert out["c"].to_list() == [7, 7, 7, 7]
        assert out["x"].dtype.is_float()  # normalized column promoted to float

    def test_mismatched_stats_length_raises(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("y", [4.0, 5.0, 6.0]))
        with pytest.raises(ValueError, match="one entry per name"):
            apply_normalize(
                df,
                ["x", "y"],
                np.asarray([0.0], dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
            )
