"""Tests for fit_imputation_mean and apply_imputation_mean."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez.step_imputation_mean import apply_imputation_mean, fit_imputation_mean


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_imputation_mean
# ---------------------------------------------------------------------------


class TestFitImputationMean:
    def test_returns_columns_and_means(self) -> None:
        columns, means = fit_imputation_mean(_df(("a", [1.0, None, 3.0])))
        assert columns == ["a"]
        assert means.dtype == np.float64
        assert means[0] == pytest.approx(2.0)  # mean of [1, 3]

    def test_parallel_to_columns_in_order(self) -> None:
        df = _df(("b", [2.0, 4.0]), ("a", [10.0, None]))
        columns, means = fit_imputation_mean(df)
        assert columns == ["b", "a"]
        assert means[0] == pytest.approx(3.0)
        assert means[1] == pytest.approx(10.0)

    def test_integer_column_mean(self) -> None:
        # Mean of an all-integer column: [1, 2, 3, 4, 5] -> 3.0.
        columns, means = fit_imputation_mean(_df(("a", [1, 2, 3, 4, 5])))
        assert columns == ["a"]
        assert means.dtype == np.float64
        assert means[0] == pytest.approx(3.0)

    def test_no_columns_returns_empties(self) -> None:
        columns, means = fit_imputation_mean(
            nw.from_native(pd.DataFrame(), eager_only=True)
        )
        assert columns == []
        assert means.shape == (0,)
        assert means.dtype == np.float64

    def test_all_null_column_kept_with_nan_mean_and_warning(self) -> None:
        df = _df(("a", [1.0, 2.0]), ("empty", [None, None]))
        with pytest.warns(UserWarning, match="entirely null"):
            columns, means = fit_imputation_mean(df)
        # "empty" is NOT dropped — it stays with a NaN mean.
        assert columns == ["a", "empty"]
        assert means[0] == pytest.approx(1.5)
        assert np.isnan(means[1])

    def test_no_warning_when_no_all_null(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            columns, _ = fit_imputation_mean(_df(("a", [1.0, None, 3.0])))
        assert columns == ["a"]


# ---------------------------------------------------------------------------
# apply_imputation_mean
# ---------------------------------------------------------------------------


class TestApplyImputationMean:
    def test_nulls_filled_with_mean(self) -> None:
        df = _df(("a", [1.0, None, 3.0]))
        out = apply_imputation_mean(df, *fit_imputation_mean(df))
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_no_nulls_unchanged_values(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]))
        out = apply_imputation_mean(df, *fit_imputation_mean(df))
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, None]), ("a", [3.0, 5.0]))
        out = apply_imputation_mean(df, *fit_imputation_mean(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_unlisted_column_passed_through(self) -> None:
        df = _df(("a", [1.0, None, 3.0]), ("other", [9.0, None, 9.0]))
        columns, means = fit_imputation_mean(_df(("a", [1.0, None, 3.0])))
        out = apply_imputation_mean(df, columns, means)
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        # "other" not in the list -> untouched (null stays null).
        assert out["other"].is_null().to_list() == [False, True, False]

    def test_all_null_column_left_unchanged(self) -> None:
        # A NaN mean (entirely-null column) is skipped: the column is left as-is.
        df = _df(("a", [1.0, None, 3.0]), ("empty", [None, None, None]))
        out = apply_imputation_mean(
            df, ["a", "empty"], np.asarray([2.0, np.nan], dtype=np.float64)
        )
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        assert out["empty"].is_null().to_list() == [True, True, True]

    def test_integer_column_with_null_imputed_and_promoted_to_float(self) -> None:
        # Polars keeps an integer column with a null as Int64; the mean (2.0)
        # fills the gap and the column is promoted to Float64 (int in -> float out).
        df = nw.from_native(pl.DataFrame({"a": [1, None, 3]}), eager_only=True)
        assert df["a"].dtype == nw.Int64
        out = apply_imputation_mean(df, *fit_imputation_mean(df))
        assert out["a"].dtype == nw.Float64
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_mismatched_means_length_raises(self) -> None:
        df = _df(("a", [1.0]), ("b", [2.0]))
        with pytest.raises(ValueError, match="one entry per name"):
            apply_imputation_mean(df, ["a", "b"], np.asarray([0.0], dtype=np.float64))
