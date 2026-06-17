"""Tests for fit_impute_mode and apply_impute_mode."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez.step_impute_mode import apply_impute_mode, fit_impute_mode


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_impute_mode
# ---------------------------------------------------------------------------


class TestFitImputeMode:
    def test_string_mode(self) -> None:
        modes = fit_impute_mode(_df(("g", ["a", "a", "b", None])))
        assert modes == {"g": "a"}

    def test_mode_picks_most_frequent(self) -> None:
        # The mode is the most frequent level among several distinct ones.
        # (Uses string data: impute_mode rejects numeric dtypes at the step
        # boundary, so a numeric column never reaches fit_impute_mode.)
        modes = fit_impute_mode(_df(("g", ["a", "a", "a", "b", "c"])))
        assert modes["g"] == "a"

    def test_tie_broken_by_smallest(self) -> None:
        # "a" and "b" each appear once -> deterministic: smallest by sort order.
        modes = fit_impute_mode(_df(("g", ["b", "a"])))
        assert modes["g"] == "a"

    def test_null_not_selected_as_mode(self) -> None:
        # Even when null is the most frequent value, the mode is a real value.
        modes = fit_impute_mode(_df(("g", [None, None, "x"])))
        assert modes["g"] == "x"

    def test_mapping_in_column_order(self) -> None:
        df = _df(("b", ["y", "y", "z"]), ("a", ["q", "q", "r"]))
        modes = fit_impute_mode(df)
        assert list(modes) == ["b", "a"]
        assert modes["b"] == "y"
        assert modes["a"] == "q"

    def test_no_columns_returns_empty(self) -> None:
        modes = fit_impute_mode(nw.from_native(pd.DataFrame(), eager_only=True))
        assert modes == {}

    def test_all_null_column_kept_with_none_mode_and_warning(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = _df(
            ("g", ["a", "a", "b"]),
            ("empty", pd.Series([None, None, None], dtype="object")),
        )
        with pytest.warns(UserWarning) as record:
            modes = fit_impute_mode(df)
        assert str(record[0].message) == snapshot
        assert list(modes) == ["g", "empty"]
        assert modes["g"] == "a"
        assert modes["empty"] is None

    def test_no_warning_when_no_all_null(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            modes = fit_impute_mode(_df(("g", ["a", None, "b"])))
        assert list(modes) == ["g"]


# ---------------------------------------------------------------------------
# apply_impute_mode
# ---------------------------------------------------------------------------


class TestApplyImputeMode:
    def test_nulls_filled_with_mode(self) -> None:
        df = _df(("g", ["a", "a", "b", None]))
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].to_list() == ["a", "a", "b", "a"]

    def test_dtype_preserved_for_string(self) -> None:
        df = _df(("g", ["a", "a", None]))
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].dtype == nw.String

    def test_categorical_input_imputed_with_mode(self) -> None:
        # impute_mode accepts Categorical as well as String — a pandas
        # categorical column must have its nulls filled with the mode, and the
        # Categorical dtype must be preserved through transform.
        df = _df(("g", pd.Categorical(["a", "a", "b", None])))
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].to_list() == ["a", "a", "b", "a"]
        assert out["g"].dtype == nw.Categorical

    def test_enum_input_imputed_with_mode(self) -> None:
        # ...and Enum, exercised through polars where the dtype is native; the
        # Enum dtype must be preserved through transform.
        pl = pytest.importorskip("polars")
        series = pl.Series(["a", "a", "b", None], dtype=pl.Enum(["a", "b"]))
        df = nw.from_native(pl.DataFrame({"g": series}), eager_only=True)
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].to_list() == ["a", "a", "b", "a"]
        assert out["g"].dtype == nw.Enum(["a", "b"])

    def test_null_most_frequent_still_imputed_with_real_level(self) -> None:
        # null is the most common value (2 nulls vs 1 "x"); imputation must still
        # fill with the real mode "x", never leave (or fill with) null.
        df = _df(("g", [None, None, "x"]))
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].to_list() == ["x", "x", "x"]

    def test_no_nulls_unchanged(self) -> None:
        df = _df(("g", ["a", "b", "c"]))
        out = apply_impute_mode(df, fit_impute_mode(df))
        assert out["g"].to_list() == ["a", "b", "c"]

    def test_unlisted_column_passed_through(self) -> None:
        df = _df(("g", ["a", None]), ("other", ["p", None]))
        modes = fit_impute_mode(_df(("g", ["a", None])))
        out = apply_impute_mode(df, modes)
        assert out["g"].to_list() == ["a", "a"]
        assert out["other"].is_null().to_list() == [False, True]

    def test_all_null_column_left_unchanged(self) -> None:
        df = _df(
            ("g", ["a", "a", None]),
            ("empty", pd.Series([None, None, None], dtype="object")),
        )
        out = apply_impute_mode(df, {"g": "a", "empty": None})
        assert out["g"].to_list() == ["a", "a", "a"]
        assert out["empty"].is_null().to_list() == [True, True, True]

    def test_empty_mapping_is_passthrough(self) -> None:
        df = _df(("g", ["a", None]), ("h", ["b", None]))
        out = apply_impute_mode(df, {})
        assert out["g"].is_null().to_list() == [False, True]
        assert out["h"].is_null().to_list() == [False, True]
