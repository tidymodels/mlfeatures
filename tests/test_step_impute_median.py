"""Tests for fit_impute_median and apply_impute_median."""

from __future__ import annotations

import math
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez.step_impute_median import (
    apply_impute_median,
    fit_impute_median,
)


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_impute_median
# ---------------------------------------------------------------------------


class TestFitImputeMedian:
    def test_returns_mapping_of_medians(self) -> None:
        # median of [1, 3, 100] (ignoring null) = 3.
        medians = fit_impute_median(_df(("a", [1.0, None, 3.0, 100.0])))
        assert list(medians) == ["a"]
        assert medians["a"] == pytest.approx(3.0)

    def test_median_differs_from_mean_on_skew(self) -> None:
        # Robust to the outlier: median 2.5, whereas the mean would be ~26.5.
        medians = fit_impute_median(_df(("a", [1.0, 2.0, 3.0, 100.0])))
        assert medians["a"] == pytest.approx(2.5)  # median of 1,2,3,100

    def test_mapping_in_column_order(self) -> None:
        df = _df(("b", [2.0, 4.0, 6.0]), ("a", [10.0, None, 30.0]))
        medians = fit_impute_median(df)
        assert list(medians) == ["b", "a"]
        assert medians["b"] == pytest.approx(4.0)
        assert medians["a"] == pytest.approx(20.0)

    def test_integer_column_median(self) -> None:
        # Median of an all-integer column: [1, 2, 3, 4, 5] -> 3.
        medians = fit_impute_median(_df(("a", [1, 2, 3, 4, 5])))
        assert list(medians) == ["a"]
        assert medians["a"] == pytest.approx(3.0)

    def test_integer_column_even_count_median(self) -> None:
        # Even count interpolates: median of [1, 2, 3, 4] -> 2.5.
        medians = fit_impute_median(_df(("a", [1, 2, 3, 4])))
        assert medians["a"] == pytest.approx(2.5)

    def test_no_columns_returns_empty(self) -> None:
        medians = fit_impute_median(nw.from_native(pd.DataFrame(), eager_only=True))
        assert medians == {}

    def test_all_null_column_kept_with_nan_median_and_warning(self) -> None:
        df = _df(
            ("a", [1.0, 2.0, 3.0]),
            ("empty", pd.Series([None, None, None], dtype="float64")),
        )
        with pytest.warns(UserWarning, match="entirely null"):
            medians = fit_impute_median(df)
        # "empty" is NOT dropped — it stays with a NaN median.
        assert list(medians) == ["a", "empty"]
        assert medians["a"] == pytest.approx(2.0)
        assert math.isnan(medians["empty"])

    def test_no_warning_when_no_all_null(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            medians = fit_impute_median(_df(("a", [1.0, None, 3.0])))
        assert list(medians) == ["a"]


# ---------------------------------------------------------------------------
# apply_impute_median
# ---------------------------------------------------------------------------


class TestApplyImputeMedian:
    def test_nulls_filled_with_median(self) -> None:
        df = _df(("a", [1.0, None, 3.0, 100.0]))
        out = apply_impute_median(df, fit_impute_median(df))
        assert out["a"].to_list() == [1.0, 3.0, 3.0, 100.0]

    def test_no_nulls_unchanged_values(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]))
        out = apply_impute_median(df, fit_impute_median(df))
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, None]), ("a", [3.0, 5.0]))
        out = apply_impute_median(df, fit_impute_median(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_unlisted_column_passed_through(self) -> None:
        df = _df(("a", [1.0, None, 3.0]), ("other", [9.0, None, 9.0]))
        medians = fit_impute_median(_df(("a", [1.0, None, 3.0])))
        out = apply_impute_median(df, medians)
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        assert out["other"].is_null().to_list() == [False, True, False]

    def test_all_null_column_left_unchanged(self) -> None:
        # A NaN median (entirely-null column) is skipped: column is left as-is.
        df = _df(
            ("a", [1.0, None, 3.0]),
            ("empty", pd.Series([None, None, None], dtype="float64")),
        )
        out = apply_impute_median(df, {"a": 2.0, "empty": math.nan})
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        assert out["empty"].is_null().to_list() == [True, True, True]

    def test_integer_column_with_null_imputed_and_promoted_to_float(self) -> None:
        # Polars keeps an integer column with a null as Int64; median (2.0)
        # fills the gap and the column is promoted to Float64 (int in -> float out).
        df = nw.from_native(pl.DataFrame({"a": [1, None, 3]}), eager_only=True)
        assert df["a"].dtype == nw.Int64
        out = apply_impute_median(df, fit_impute_median(df))
        assert out["a"].dtype == nw.Float64
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_empty_mapping_is_passthrough(self) -> None:
        df = _df(("a", [1.0, None]), ("b", [2.0, None]))
        out = apply_impute_median(df, {})
        assert out["a"].is_null().to_list() == [False, True]
        assert out["b"].is_null().to_list() == [False, True]
