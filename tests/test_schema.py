"""Tests for the column-lineage DAG (issue #19).

This module covers the structure and its initial construction (`Schema`,
`ColumnNode`, `Schema.from_frame`). The ``update()`` application logic,
``summary()`` and provenance queries are covered as they land.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
import polars as pl
import pytest

from shoez import _frame
from shoez.schema import ColumnNode, Schema
from shoez.types import Roles

Constructor = Callable[[dict[str, Any]], Any]

BACKENDS: dict[str, Constructor] = {"pandas": pd.DataFrame, "polars": pl.DataFrame}


def _frame_of(backend: str, data: dict[str, Any]) -> Any:
    """Wrap a native frame built by ``backend`` as an internal narwhals frame."""
    return _frame.from_native(BACKENDS[backend](data))


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_from_frame_builds_one_original_node_per_column(backend: str) -> None:
    frame = _frame_of(
        backend, {"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]}
    )
    schema = Schema.from_frame(frame)
    expected_dtypes = _frame.dtypes(frame)

    assert schema.columns == ["a", "b", "c"]
    for name in ("a", "b", "c"):
        node = schema[name]
        assert node.name == name
        assert node.dtype == expected_dtypes[name]
        assert node.source == "original"
        assert node.parent_cols == []
        assert node.dtype_history == [expected_dtypes[name]]
        assert node.sparse is False
        assert node.removed_by is None


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_from_frame_pulls_roles(backend: str) -> None:
    frame = _frame_of(backend, {"y": [0, 1, 0], "x": [1.0, 2.0, 3.0]})
    roles = Roles({"y": {"outcome"}, "x": {"predictor"}})
    schema = Schema.from_frame(frame, roles)

    assert schema["y"].roles == {"outcome"}
    assert schema["x"].roles == {"predictor"}


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_from_frame_without_roles_gives_empty_role_sets(backend: str) -> None:
    schema = Schema.from_frame(_frame_of(backend, {"a": [1, 2, 3]}))
    assert schema["a"].roles == set()


def test_node_roles_are_independent_copies() -> None:
    """Mutating a node's roles must not bleed back into the source Roles."""
    frame = _frame_of("polars", {"a": [1, 2, 3]})
    roles = Roles({"a": {"predictor"}})
    schema = Schema.from_frame(frame, roles)

    schema["a"].roles.add("id")
    assert roles.of("a") == {"predictor"}


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_columns_and_dtypes_views(backend: str) -> None:
    frame = _frame_of(backend, {"a": [1, 2, 3], "b": ["x", "y", "z"]})
    schema = Schema.from_frame(frame)

    assert schema.columns == ["a", "b"]
    assert schema.dtypes == _frame.dtypes(frame)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_mapping_protocol(backend: str) -> None:
    frame = _frame_of(backend, {"a": [1, 2, 3], "b": [4, 5, 6]})
    schema = Schema.from_frame(frame)

    assert len(schema) == 2
    assert "a" in schema
    assert "missing" not in schema
    assert isinstance(schema["a"], ColumnNode)
    with pytest.raises(KeyError):
        schema["missing"]


def test_empty_frame_yields_empty_schema() -> None:
    schema = Schema.from_frame(_frame_of("polars", {}))
    assert schema.columns == []
    assert len(schema) == 0


def test_fresh_schema_has_no_history() -> None:
    schema = Schema.from_frame(_frame_of("polars", {"a": [1, 2, 3]}))
    assert schema._history == []
