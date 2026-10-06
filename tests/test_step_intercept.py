"""Tests for add_intercept (the apply half of the intercept step)."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl

from mlfeatures.step_intercept import add_intercept


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)))


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_narwhals_frame(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0, 3.0])))
        assert isinstance(result, nw.DataFrame)

    def test_new_column_is_present(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0, 3.0])))
        assert "intercept" in result.columns


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


class TestCoreBehavior:
    def test_default_adds_column_of_ones(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0, 3.0])))
        assert result["intercept"].to_list() == [1.0, 1.0, 1.0]

    def test_custom_value(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0])), value=0.0)
        assert result["intercept"].to_list() == [0.0, 0.0]

    def test_custom_name(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0])), name="b0")
        assert "b0" in result.columns
        assert "intercept" not in result.columns

    def test_existing_columns_untouched(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0]), ("y", [3.0, 4.0])))
        assert result["x"].to_list() == [1.0, 2.0]
        assert result["y"].to_list() == [3.0, 4.0]

    def test_new_column_appended_last(self) -> None:
        result = add_intercept(_df(("x", [1.0]), ("y", [2.0])))
        assert result.columns == ["x", "y", "intercept"]


# ---------------------------------------------------------------------------
# Edge shapes
# ---------------------------------------------------------------------------


class TestEdgeShapes:
    def test_single_row(self) -> None:
        result = add_intercept(_df(("x", [7.0])))
        assert result["intercept"].to_list() == [1.0]

    def test_zero_rows(self) -> None:
        # The literal broadcasts to length 0: the column exists but is empty.
        df = nw.from_native(pl.DataFrame({"x": []}, schema={"x": pl.Float64}))
        result = add_intercept(df)
        assert "intercept" in result.columns
        assert result["intercept"].to_list() == []

    def test_zero_columns(self) -> None:
        # The intercept is independent of input columns: a frame with no columns
        # but N rows still gets an N-length column of ones.
        df = nw.from_native(pd.DataFrame(index=range(3)))
        result = add_intercept(df)
        assert result.columns == ["intercept"]
        assert result["intercept"].to_list() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class TestBackends:
    def test_polars_in_polars_out(self) -> None:
        df = nw.from_native(pl.DataFrame({"x": [1.0, 2.0, 3.0]}))
        result = add_intercept(df, value=2.0)
        assert isinstance(result.to_native(), pl.DataFrame)
        assert result["intercept"].to_list() == [2.0, 2.0, 2.0]

    def test_pandas_in_pandas_out(self) -> None:
        result = add_intercept(_df(("x", [1.0, 2.0, 3.0])))
        assert isinstance(result.to_native(), pd.DataFrame)


# ---------------------------------------------------------------------------
# Name clash
# ---------------------------------------------------------------------------


class TestNameClash:
    """``add_intercept`` itself does not guard against a clash.

    Name-collision rejection is centralized in the recipe
    (``Schema.check_additions``); see the recipe-level test in
    ``test_step_intercept_e2e.py``. The bare helper just overwrites.
    """

    def test_existing_name_is_overwritten_not_an_error(self) -> None:
        df = _df(("intercept", [1.0, 2.0]))
        result = add_intercept(df, value=7.0)
        assert result.to_native()["intercept"].tolist() == [7.0, 7.0]
