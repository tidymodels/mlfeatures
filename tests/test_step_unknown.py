"""Tests for fit_unknown and apply_unknown."""

from __future__ import annotations

import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_unknown import apply_unknown, fit_unknown


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# fit_unknown
# ---------------------------------------------------------------------------


class TestFitUnknown:
    def test_returns_all_columns_in_order(self) -> None:
        df = _df(("b", ["x", None]), ("a", ["y", "z"]))
        assert fit_unknown(df) == ["b", "a"]

    def test_columns_without_nulls_still_returned(self) -> None:
        # parity: a column with no training nulls is still recoded (new data may
        # have nulls), so it must be in the returned list.
        df = _df(("g", ["a", "b", "c"]))
        assert fit_unknown(df) == ["g"]

    def test_collision_warns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("g", ["a", "unknown", None]))
        with pytest.warns(UserWarning) as record:
            result = fit_unknown(df)
        assert str(record[0].message) == snapshot
        assert result == ["g"]

    def test_no_collision_no_warning(self) -> None:
        df = _df(("g", ["a", "b", None]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_unknown(df) == ["g"]

    def test_custom_level_collision(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("g", ["a", "MISSING", None]))
        with pytest.warns(UserWarning) as record:
            fit_unknown(df, new_level="MISSING")
        assert str(record[0].message) == snapshot


# ---------------------------------------------------------------------------
# apply_unknown
# ---------------------------------------------------------------------------


class TestApplyUnknown:
    def test_nulls_replaced_with_unknown(self) -> None:
        df = _df(("g", ["a", "b", None]))
        out = apply_unknown(df, ["g"])
        assert out["g"].to_list() == ["a", "b", "unknown"]

    def test_non_null_values_unchanged(self) -> None:
        df = _df(("g", ["a", "b", "c"]))
        out = apply_unknown(df, ["g"])
        assert out["g"].to_list() == ["a", "b", "c"]

    def test_custom_level(self) -> None:
        df = _df(("g", ["a", None]))
        out = apply_unknown(df, ["g"], new_level="MISSING")
        assert out["g"].to_list() == ["a", "MISSING"]

    def test_output_dtype_is_string(self) -> None:
        df = _df(("g", pd.Categorical(["a", None, "a"])))
        out = apply_unknown(df, ["g"])
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "unknown", "a"]

    def test_string_input_recoded_to_string(self) -> None:
        # unknown accepts String as well as Categorical/Enum; a string column's
        # nulls are recoded and the output stays String.
        df = _df(("g", ["a", "b", None]))
        out = apply_unknown(df, ["g"])
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "b", "unknown"]

    def test_enum_input_recoded_to_string(self) -> None:
        # ...and Enum, exercised through polars where the dtype is native.
        pl = pytest.importorskip("polars")
        series = pl.Series(["a", None, "a"], dtype=pl.Enum(["a", "b"]))
        df = nw.from_native(pl.DataFrame({"g": series}), eager_only=True)
        out = apply_unknown(df, ["g"])
        assert out["g"].dtype == nw.String
        assert out["g"].to_list() == ["a", "unknown", "a"]

    def test_columns_not_listed_pass_through(self) -> None:
        df = _df(("g", ["a", None]), ("other", ["p", None]))
        out = apply_unknown(df, ["g"])
        assert out["g"].to_list() == ["a", "unknown"]
        # "other" was not in the list -> untouched (its null stays null).
        assert out["other"].is_null().to_list() == [False, True]
        assert out["other"][0] == "p"

    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", ["x", None]), ("a", ["y", "z"]))
        out = apply_unknown(df, ["b", "a"])
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape
