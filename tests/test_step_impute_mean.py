"""Tests for fit_impute_mean and apply_impute_mean."""

from __future__ import annotations

import math
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_impute_mean import apply_impute_mean, fit_impute_mean


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_impute_mean
# ---------------------------------------------------------------------------


class TestFitImputeMean:
    def test_returns_mapping_of_means(self) -> None:
        means = fit_impute_mean(_df(("a", [1.0, None, 3.0])))
        assert list(means) == ["a"]
        assert means["a"] == pytest.approx(2.0)  # mean of [1, 3]

    def test_mapping_in_column_order(self) -> None:
        df = _df(("b", [2.0, 4.0]), ("a", [10.0, None]))
        means = fit_impute_mean(df)
        assert list(means) == ["b", "a"]
        assert means["b"] == pytest.approx(3.0)
        assert means["a"] == pytest.approx(10.0)

    def test_integer_column_mean(self) -> None:
        # Mean of an all-integer column: [1, 2, 3, 4, 5] -> 3.0.
        means = fit_impute_mean(_df(("a", [1, 2, 3, 4, 5])))
        assert list(means) == ["a"]
        assert means["a"] == pytest.approx(3.0)

    def test_no_columns_returns_empty(self) -> None:
        means = fit_impute_mean(nw.from_native(pd.DataFrame(), eager_only=True))
        assert means == {}

    def test_all_null_column_kept_with_nan_mean_and_warning(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = _df(("a", [1.0, 2.0]), ("empty", [None, None]))
        with pytest.warns(UserWarning) as record:
            means = fit_impute_mean(df)
        # numpy may also emit a RuntimeWarning ("Mean of empty slice"); select
        # our UserWarning explicitly rather than relying on record order.
        assert str(record.pop(UserWarning).message) == snapshot
        # "empty" is NOT dropped: it stays with a NaN mean.
        assert list(means) == ["a", "empty"]
        assert means["a"] == pytest.approx(1.5)
        assert math.isnan(means["empty"])

    def test_no_warning_when_no_all_null(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            means = fit_impute_mean(_df(("a", [1.0, None, 3.0])))
        assert list(means) == ["a"]


# ---------------------------------------------------------------------------
# apply_impute_mean
# ---------------------------------------------------------------------------


class TestApplyImputeMean:
    def test_nulls_filled_with_mean(self) -> None:
        df = _df(("a", [1.0, None, 3.0]))
        out = apply_impute_mean(df, fit_impute_mean(df))
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_no_nulls_unchanged_values(self) -> None:
        df = _df(("a", [1.0, 2.0, 3.0]))
        out = apply_impute_mean(df, fit_impute_mean(df))
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, None]), ("a", [3.0, 5.0]))
        out = apply_impute_mean(df, fit_impute_mean(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_unlisted_column_passed_through(self) -> None:
        df = _df(("a", [1.0, None, 3.0]), ("other", [9.0, None, 9.0]))
        means = fit_impute_mean(_df(("a", [1.0, None, 3.0])))
        out = apply_impute_mean(df, means)
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        # "other" not in the mapping -> untouched (null stays null).
        assert out["other"].is_null().to_list() == [False, True, False]

    def test_all_null_column_left_unchanged(self) -> None:
        # A NaN mean (entirely-null column) is skipped: the column is left as-is.
        df = _df(("a", [1.0, None, 3.0]), ("empty", [None, None, None]))
        out = apply_impute_mean(df, {"a": 2.0, "empty": math.nan})
        assert out["a"].to_list() == [1.0, 2.0, 3.0]
        assert out["empty"].is_null().to_list() == [True, True, True]

    def test_integer_column_with_null_imputed_and_promoted_to_float(self) -> None:
        # Polars keeps an integer column with a null as Int64; the mean (2.0)
        # fills the gap and the column is promoted to Float64 (int in -> float out).
        df = nw.from_native(pl.DataFrame({"a": [1, None, 3]}), eager_only=True)
        assert df["a"].dtype == nw.Int64
        out = apply_impute_mean(df, fit_impute_mean(df))
        assert out["a"].dtype == nw.Float64
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_empty_mapping_is_passthrough(self) -> None:
        df = _df(("a", [1.0, None]), ("b", [2.0, None]))
        out = apply_impute_mean(df, {})
        assert out["a"].is_null().to_list() == [False, True]
        assert out["b"].is_null().to_list() == [False, True]
