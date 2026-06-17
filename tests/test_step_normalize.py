"""Tests for fit_normalize and apply_normalize."""

from __future__ import annotations

import math
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez.step_normalize import apply_normalize, fit_normalize


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_mapping_of_mean_std_pairs(self) -> None:
        stats = fit_normalize(_df(("a", [1.0, 2.0, 3.0])))
        assert list(stats) == ["a"]
        mean, std = stats["a"]
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_one_entry_per_column(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]), ("b", [4.0, 5.0, 6.0]))
        stats = fit_normalize(df)
        assert list(stats) == ["a", "b"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_columns_returns_empty(self) -> None:
        stats = fit_normalize(nw.from_native(pd.DataFrame(), eager_only=True))
        assert stats == {}

    def test_zero_rows_yields_nan(self) -> None:
        df = nw.from_native(
            pd.DataFrame({"a": pd.Series([], dtype="float64")}), eager_only=True
        )
        stats = fit_normalize(df)
        # NaN std is not treated as constant — the column is kept.
        assert list(stats) == ["a"]
        mean, std = stats["a"]
        assert math.isnan(mean)
        assert math.isnan(std)

    def test_single_row_yields_nan_std(self) -> None:
        stats = fit_normalize(_df(("a", [42.0])))
        assert list(stats) == ["a"]
        mean, std = stats["a"]
        assert mean == 42.0
        assert math.isnan(std)


# ---------------------------------------------------------------------------
# Core estimates
# ---------------------------------------------------------------------------


class TestCoreEstimates:
    def test_known_mean_and_sample_std(self) -> None:
        # [1, 2, 3]: mean 2, sample std (ddof=1) = 1.0.
        mean, std = fit_normalize(_df(("a", [1.0, 2.0, 3.0])))["a"]
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(1.0)

    def test_column_order_preserved(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        stats = fit_normalize(df)
        assert list(stats) == ["b", "a"]
        assert stats["b"][0] == pytest.approx(2.0)
        assert stats["a"][0] == pytest.approx(20.0)

    def test_integer_column_accepted(self) -> None:
        mean, std = fit_normalize(_df(("a", [1, 2, 3])))["a"]
        assert mean == pytest.approx(2.0)
        assert std == pytest.approx(1.0)

    def test_normalized_column_has_unit_std(self) -> None:
        # The invariant: scaling by the returned (ddof=1) std must make the
        # normalized column's sample std exactly 1.
        col = _df(("a", [3.0, 7.0, 11.0, 2.0, 9.0]))
        mean, std = fit_normalize(col)["a"]
        normalized = (col["a"] - mean) / std
        assert normalized.std(ddof=1) == pytest.approx(1.0)
        assert normalized.mean() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Zero-variance (constant) columns are dropped with a warning
# ---------------------------------------------------------------------------


class TestConstantColumns:
    def test_constant_column_is_omitted(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning) as record:
            stats = fit_normalize(df)
        assert str(record[0].message) == snapshot
        assert list(stats) == ["x"]  # constant "c" dropped

    def test_warning_names_the_columns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning) as record:
            stats = fit_normalize(df)
        assert str(record[0].message) == snapshot
        assert stats == {}

    def test_no_warning_when_no_constant_columns(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stats = fit_normalize(_df(("x", [1.0, 2.0, 3.0])))
        assert list(stats) == ["x"]


# ---------------------------------------------------------------------------
# apply_normalize
# ---------------------------------------------------------------------------


class TestApplyNormalize:
    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        out = apply_normalize(df, fit_normalize(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_each_column_centered_and_unit_std(self) -> None:
        df = _df(("x", [3.0, 7.0, 11.0, 2.0, 9.0]), ("y", [-1.0, 0.0, 1.0, 5.0, 4.0]))
        out = apply_normalize(df, fit_normalize(df))
        for col in out.columns:
            assert out[col].mean() == pytest.approx(0.0)
            assert out[col].std(ddof=1) == pytest.approx(1.0)

    def test_round_trip_with_fit(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0, 4.0]))
        stats = fit_normalize(df)
        out = apply_normalize(df, stats)
        mean, std = stats["x"]
        expected = [(v - mean) / std for v in [1.0, 2.0, 3.0, 4.0]]
        assert out["x"].to_list() == pytest.approx(expected)

    def test_constant_column_passed_through_untouched(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("c", [5.0, 5.0, 5.0]))
        with pytest.warns(UserWarning):
            fitted = fit_normalize(df)
        out = apply_normalize(df, fitted)
        # "c" was dropped at fit time, so it is not normalized — left as-is.
        assert out["c"].to_list() == [5.0, 5.0, 5.0]
        assert out["x"].mean() == pytest.approx(0.0)

    def test_integer_input_yields_float_output(self) -> None:
        df = _df(("a", [1, 2, 3, 4]))
        assert df["a"].dtype.is_integer()  # integer in
        out = apply_normalize(df, fit_normalize(df))
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
            stats = fit_normalize(df)
        assert list(stats) == ["x"]  # constant "c" not normalized
        out = apply_normalize(df, stats)
        assert out["c"].dtype.is_integer()  # int in -> int out (untouched)
        assert out["c"].to_list() == [7, 7, 7, 7]
        assert out["x"].dtype.is_float()  # normalized column promoted to float

    def test_empty_stats_is_passthrough(self) -> None:
        df = _df(("x", [1.0, 2.0, 3.0]), ("y", [4.0, 5.0, 6.0]))
        out = apply_normalize(df, {})
        assert out["x"].to_list() == [1.0, 2.0, 3.0]
        assert out["y"].to_list() == [4.0, 5.0, 6.0]
