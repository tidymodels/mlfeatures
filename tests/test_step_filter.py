"""Tests for apply_filter (the apply half of the filter step) and StepFilter guards."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez.step_filter import StepFilter, apply_filter


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)))


class TestApplyFilter:
    def test_keeps_matching_rows(self) -> None:
        result = apply_filter(_df(("a", [1, 2, 3, 4])), lambda d: d["a"] > 2)
        assert result["a"].to_list() == [3, 4]

    def test_all_true_keeps_everything(self) -> None:
        result = apply_filter(_df(("a", [1, 2, 3])), lambda d: d["a"] > 0)
        assert result["a"].to_list() == [1, 2, 3]

    def test_all_false_drops_everything(self) -> None:
        result = apply_filter(_df(("a", [1, 2, 3])), lambda d: d["a"] > 99)
        assert result["a"].to_list() == []

    def test_column_schema_unchanged(self) -> None:
        result = apply_filter(
            _df(("a", [1, 2, 3]), ("b", [4, 5, 6])), lambda d: d["a"] > 1
        )
        assert result.columns == ["a", "b"]


class TestEdgeShapes:
    def test_single_row(self) -> None:
        result = apply_filter(_df(("a", [5])), lambda d: d["a"] > 1)
        assert result["a"].to_list() == [5]

    def test_zero_rows(self) -> None:
        df = nw.from_native(pl.DataFrame({"a": []}, schema={"a": pl.Float64}))
        result = apply_filter(df, lambda d: d["a"] > 1)
        assert result.columns == ["a"]
        assert result["a"].to_list() == []


class TestBackends:
    def test_polars_in_polars_out(self) -> None:
        df = nw.from_native(pl.DataFrame({"a": [1, 2, 3, 4]}))
        result = apply_filter(df, lambda d: d["a"] > 2)
        assert isinstance(result.to_native(), pl.DataFrame)
        assert result["a"].to_list() == [3, 4]

    def test_pandas_in_pandas_out(self) -> None:
        result = apply_filter(_df(("a", [1, 2, 3, 4])), lambda d: d["a"] > 2)
        assert isinstance(result.to_native(), pd.DataFrame)


class TestConstructorValidation:
    def test_non_callable_predicate_raises(self) -> None:
        with pytest.raises(TypeError):
            StepFilter("a > 1")  # type: ignore[arg-type]

    def test_non_callable_predicate_message(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(TypeError) as excinfo:
            StepFilter("a > 1")  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot

    def test_default_id(self) -> None:
        assert StepFilter(lambda d: d["a"] > 1).id == "filter"

    def test_custom_id(self) -> None:
        assert StepFilter(lambda d: d["a"] > 1, id="keep_big").id == "keep_big"
