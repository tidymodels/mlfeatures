"""Tests for apply_mutate (the apply half of the mutate step) and StepMutate guards."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from shoez.step_mutate import StepMutate, apply_mutate


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)))


class TestApplyMutate:
    def test_adds_new_column(self) -> None:
        result = apply_mutate(_df(("x", [1, 2, 3])), {"y": lambda d: d["x"] * 2})
        assert result.columns == ["x", "y"]
        assert result["y"].to_list() == [2, 4, 6]

    def test_overwrites_existing_column_in_place(self) -> None:
        result = apply_mutate(_df(("x", [1, 2, 3])), {"x": lambda d: d["x"] + 10})
        assert result.columns == ["x"]
        assert result["x"].to_list() == [11, 12, 13]

    def test_multiple_mutations_see_original_frame(self) -> None:
        # Both expressions read the *original* x, matching with_columns semantics.
        result = apply_mutate(
            _df(("x", [1, 2, 3])),
            {"x": lambda d: d["x"] + 1, "x_sq": lambda d: d["x"] ** 2},
        )
        assert result["x"].to_list() == [2, 3, 4]
        assert result["x_sq"].to_list() == [1, 4, 9]


class TestEdgeShapes:
    def test_single_row(self) -> None:
        result = apply_mutate(_df(("x", [7])), {"y": lambda d: d["x"] * 2})
        assert result["y"].to_list() == [14]

    def test_zero_rows(self) -> None:
        df = nw.from_native(pl.DataFrame({"x": []}, schema={"x": pl.Float64}))
        result = apply_mutate(df, {"y": lambda d: d["x"] * 2})
        assert result.columns == ["x", "y"]
        assert result["y"].to_list() == []


class TestBackends:
    def test_polars_in_polars_out(self) -> None:
        df = nw.from_native(pl.DataFrame({"x": [1, 2, 3]}))
        result = apply_mutate(df, {"y": lambda d: d["x"] * 2})
        assert isinstance(result.to_native(), pl.DataFrame)
        assert result["y"].to_list() == [2, 4, 6]

    def test_pandas_in_pandas_out(self) -> None:
        result = apply_mutate(_df(("x", [1, 2, 3])), {"y": lambda d: d["x"] * 2})
        assert isinstance(result.to_native(), pd.DataFrame)


class TestConstructorValidation:
    def test_empty_mapping_raises(self) -> None:
        with pytest.raises(TypeError):
            StepMutate({})

    def test_non_callable_mutation_raises(self) -> None:
        with pytest.raises(TypeError):
            StepMutate({"y": 5})  # type: ignore[dict-item]

    def test_default_id(self) -> None:
        assert StepMutate({"y": lambda d: d["x"]}).id == "mutate"

    def test_default_output_role_is_predictor(self) -> None:
        assert StepMutate({"y": lambda d: d["x"]}).output_role == "predictor"
