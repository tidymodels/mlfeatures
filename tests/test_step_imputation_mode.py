"""Tests for fit_imputation_mode and apply_imputation_mode."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest

from shoez.step_imputation_mode import apply_imputation_mode, fit_imputation_mode


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_imputation_mode
# ---------------------------------------------------------------------------


class TestFitImputationMode:
    def test_string_mode(self) -> None:
        columns, modes = fit_imputation_mode(_df(("g", ["a", "a", "b", None])))
        assert columns == ["g"]
        assert modes == ["a"]

    def test_numeric_mode(self) -> None:
        _, modes = fit_imputation_mode(_df(("n", [1, 1, 1, 2, 3])))
        assert modes[0] == 1

    def test_tie_broken_by_smallest(self) -> None:
        # "a" and "b" each appear once -> deterministic: smallest by sort order.
        _, modes = fit_imputation_mode(_df(("g", ["b", "a"])))
        assert modes[0] == "a"

    def test_null_not_selected_as_mode(self) -> None:
        # Even when null is the most frequent value, the mode is a real value.
        _, modes = fit_imputation_mode(_df(("g", [None, None, "x"])))
        assert modes[0] == "x"

    def test_parallel_to_columns_in_order(self) -> None:
        df = _df(("b", ["y", "y", "z"]), ("a", [5, 5, 9]))
        columns, modes = fit_imputation_mode(df)
        assert columns == ["b", "a"]
        assert modes[0] == "y"
        assert modes[1] == 5

    def test_no_columns_returns_empties(self) -> None:
        columns, modes = fit_imputation_mode(
            nw.from_native(pd.DataFrame(), eager_only=True)
        )
        assert columns == []
        assert modes == []

    def test_all_null_column_kept_with_none_mode_and_warning(self) -> None:
        df = _df(
            ("g", ["a", "a", "b"]),
            ("empty", pd.Series([None, None, None], dtype="object")),
        )
        with pytest.warns(UserWarning, match="entirely null"):
            columns, modes = fit_imputation_mode(df)
        assert columns == ["g", "empty"]
        assert modes[0] == "a"
        assert modes[1] is None

    def test_no_warning_when_no_all_null(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            columns, _ = fit_imputation_mode(_df(("g", ["a", None, "b"])))
        assert columns == ["g"]


# ---------------------------------------------------------------------------
# apply_imputation_mode
# ---------------------------------------------------------------------------


class TestApplyImputationMode:
    def test_nulls_filled_with_mode(self) -> None:
        df = _df(("g", ["a", "a", "b", None]))
        out = apply_imputation_mode(df, *fit_imputation_mode(df))
        assert out["g"].to_list() == ["a", "a", "b", "a"]

    def test_dtype_preserved_for_string(self) -> None:
        df = _df(("g", ["a", "a", None]))
        out = apply_imputation_mode(df, *fit_imputation_mode(df))
        assert out["g"].dtype == nw.String

    def test_no_nulls_unchanged(self) -> None:
        df = _df(("g", ["a", "b", "c"]))
        out = apply_imputation_mode(df, *fit_imputation_mode(df))
        assert out["g"].to_list() == ["a", "b", "c"]

    def test_unlisted_column_passed_through(self) -> None:
        df = _df(("g", ["a", None]), ("other", ["p", None]))
        columns, modes = fit_imputation_mode(_df(("g", ["a", None])))
        out = apply_imputation_mode(df, columns, modes)
        assert out["g"].to_list() == ["a", "a"]
        assert out["other"].is_null().to_list() == [False, True]

    def test_all_null_column_left_unchanged(self) -> None:
        df = _df(
            ("g", ["a", "a", None]),
            ("empty", pd.Series([None, None, None], dtype="object")),
        )
        out = apply_imputation_mode(df, ["g", "empty"], ["a", None])
        assert out["g"].to_list() == ["a", "a", "a"]
        assert out["empty"].is_null().to_list() == [True, True, True]

    def test_mismatched_modes_length_raises(self) -> None:
        df = _df(("a", ["x"]), ("b", ["y"]))
        with pytest.raises(ValueError, match="one entry per name"):
            apply_imputation_mode(df, ["a", "b"], ["x"])
