"""Tests for the column-lineage DAG (issue #19).

This module covers the structure and its initial construction (`Schema`,
`ColumnNode`, `Schema.from_frame`). The ``update()`` application logic,
``summary()`` and provenance queries are covered as they land.
"""

from __future__ import annotations

from typing import Any, Callable

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import (
    ANY_DTYPE,
    ColSchema,
    Mode,
    Step,
    StepClassDistance,
    StepDropZeroVariance,
    StepDummy,
    StepIntercept,
    StepNormalize,
    _frame,
)
from shoez.schema import ColumnNode, Schema
from shoez.types import AcceptedDtypes, Frame, Roles, SchemaInfo

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


# --- Schema.update -----------------------------------------------------------


class _FakeReduce(Step):
    """Synthetic ``MANY_TO_MANY`` step: drops its inputs, adds constant columns.

    The catalog has no ``MANY_TO_MANY`` step yet (no PCA), so this stand-in
    exercises the all-inputs-to-every-output provenance path in ``update()``.
    ``parent_override`` lets a test feed a deliberately wrong ``output_parents``
    to drive the integrity guards.
    """

    mode = Mode.MANY_TO_MANY

    def __init__(
        self,
        columns: list[str],
        out_names: tuple[str, ...] = ("pc1", "pc2"),
        *,
        parent_override: dict[str, list[str]] | None = None,
        id: str = "reduce",
    ) -> None:
        self.columns = columns
        self.out_names = list(out_names)
        self._parent_override = parent_override
        self.id = id
        self.trained = False
        self.keep_original_cols = False
        self.output_role = "predictor"
        self._inputs: list[str] = []

    def fit(self, training_data: Frame, info: SchemaInfo) -> "_FakeReduce":
        self._inputs = _frame.select(training_data, self.columns, info.roles).columns
        return self

    def transform(self, new_data: Frame) -> Frame:
        exprs = [nw.lit(float(i)).alias(n) for i, n in enumerate(self.out_names, 1)]
        return _frame.drop(new_data.with_columns(exprs), self._inputs)

    def schema_in(self) -> ColSchema:
        return ColSchema(columns=self._inputs)

    def schema_out(self) -> ColSchema:
        return ColSchema(columns=self.out_names)

    def output_parents(self) -> dict[str, list[str]]:
        if self._parent_override is not None:
            return self._parent_override
        return {name: list(self._inputs) for name in self.out_names}

    def accepts_dtypes(self) -> AcceptedDtypes:
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        return False


def _apply(
    steps: list[Step], backend: str, data: dict[str, Any], roles: Roles | None = None
) -> tuple[Schema, Frame]:
    """Run ``steps`` through the recipe's per-step sequence, returning the DAG.

    Mirrors the loop the recipe will run in commit 5 (fit, transform,
    update_roles, schema.update) so ``update()`` can be tested in isolation.
    """
    roles = roles if roles is not None else Roles()
    frame = _frame_of(backend, data)
    schema = Schema.from_frame(frame, roles)
    for step in steps:
        fitted = step.fit(frame, SchemaInfo(roles=roles))
        frame = fitted.transform(frame)
        fitted.update_roles(roles)
        schema.update(fitted, frame, roles)
    return schema, frame


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_one_to_many_dummy_attributes_each_output_to_its_source(
    backend: str,
) -> None:
    schema, _ = _apply(
        [StepDummy(["g"])], backend, {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}
    )
    assert "g" not in schema  # source column replaced
    node = schema["g_b"]
    assert node.source == "dummy"
    assert node.parent_cols == ["g"]
    archived = [n for n in schema._history if n.name == "g"]
    assert len(archived) == 1
    assert archived[0].removed_by == "dummy"


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_one_to_one_records_dtype_change_not_source(backend: str) -> None:
    schema, _ = _apply(
        [StepNormalize(["a"])], backend, {"a": [1, 2, 3], "g": ["x", "y", "x"]}
    )
    node = schema["a"]
    # In-place transform: the column is not "produced", so source is unchanged.
    assert node.source == "original"
    assert node.dtype_history[0] == nw.Int64
    assert node.dtype_history[-1] == nw.Float64
    assert len(node.dtype_history) == 2


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_removing_archives_dropped_column(backend: str) -> None:
    schema, _ = _apply(
        [StepDropZeroVariance(["a", "const"])],
        backend,
        {"a": [1.0, 2.0, 3.0], "const": [7.0, 7.0, 7.0]},
    )
    assert schema.columns == ["a"]
    archived = [n for n in schema._history if n.name == "const"]
    assert len(archived) == 1
    assert archived[0].removed_by == "zv"


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_adding_constant_has_no_parents(backend: str) -> None:
    schema, _ = _apply([StepIntercept()], backend, {"a": [1.0, 2.0, 3.0]})
    node = schema["intercept"]
    assert node.source == "intercept"
    assert node.parent_cols == []


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_adding_derived_has_feature_parents(backend: str) -> None:
    schema, _ = _apply(
        [StepClassDistance(["a", "b"], outcome="out")],
        backend,
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "out": [0, 0, 1, 1]},
    )
    produced = [n for n in schema.columns if n.startswith("distance_")]
    assert len(produced) == 2
    for name in produced:
        assert schema[name].source == "class_distance"
        assert schema[name].parent_cols == ["a", "b"]


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_update_many_to_many_every_output_descends_from_all_inputs(
    backend: str,
) -> None:
    schema, _ = _apply(
        [_FakeReduce(["a", "b", "c"])],
        backend,
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]},
    )
    assert schema.columns == ["pc1", "pc2"]
    for name in ("pc1", "pc2"):
        assert schema[name].source == "reduce"
        assert schema[name].parent_cols == ["a", "b", "c"]
    assert {n.name for n in schema._history} == {"a", "b", "c"}


def test_update_rejects_name_collision() -> None:
    with pytest.raises(ValueError, match="already exist"):
        _apply(
            [_FakeReduce(["a", "b"], out_names=("h",))],
            "polars",
            {"a": [1, 2, 3], "b": [4, 5, 6], "h": [7, 8, 9]},
        )


def test_update_rejects_undeclared_produced_column() -> None:
    with pytest.raises(ValueError, match="do not match"):
        _apply(
            # transform produces pc1 + pc2 but output_parents declares only pc1
            [_FakeReduce(["a", "b"], parent_override={"pc1": ["a", "b"]})],
            "polars",
            {"a": [1, 2, 3], "b": [4, 5, 6]},
        )


def test_update_rejects_unknown_parent() -> None:
    with pytest.raises(ValueError, match="not in the schema"):
        _apply(
            [_FakeReduce(["a", "b"], parent_override={"pc1": ["ghost"], "pc2": ["a"]})],
            "polars",
            {"a": [1, 2, 3], "b": [4, 5, 6]},
        )


def test_update_propagates_roles_to_produced_columns() -> None:
    roles = Roles({"g": {"predictor"}})
    schema, _ = _apply(
        [StepDummy(["g"])], "polars", {"g": ["a", "b", "a"]}, roles=roles
    )
    assert schema["g_b"].roles == {"predictor"}


def test_update_composes_across_a_multi_step_sequence() -> None:
    schema, frame = _apply(
        [StepNormalize(["a"]), StepDummy(["g"])],
        "pandas",
        {"a": [1, 2, 3], "g": ["x", "y", "x"]},
    )
    # normalize left `a` in place (rescaled to float); dummy replaced `g`.
    assert schema["a"].source == "original"
    assert schema["a"].dtype_history[-1] == nw.Float64
    assert "g" not in schema
    assert schema["g_y"].source == "dummy"
    assert schema["g_y"].parent_cols == ["g"]
    assert any(n.name == "g" and n.removed_by == "dummy" for n in schema._history)
    # the live DAG order matches the transformed frame.
    assert schema.columns == _frame.columns(frame)


# --- summary() and provenance queries ----------------------------------------


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_summary_columns_and_values(backend: str) -> None:
    frame = _frame_of(backend, {"a": [1, 2, 3], "b": [1.5, 2.5, 3.5]})
    schema = Schema.from_frame(frame, Roles({"a": {"predictor"}}))
    out = nw.from_native(schema.summary())

    assert out.columns == ["name", "dtype", "roles", "source", "sparse"]
    assert out["name"].to_list() == ["a", "b"]
    assert out["dtype"].to_list() == ["int", "float"]  # taxonomy categories
    assert out["roles"].to_list() == ["predictor", ""]
    assert out["source"].to_list() == ["original", "original"]
    assert out["sparse"].to_list() == [False, False]


def test_summary_reflects_steps() -> None:
    schema, _ = _apply(
        [StepDummy(["g"])], "polars", {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}
    )
    out = nw.from_native(schema.summary())
    source_of = dict(zip(out["name"].to_list(), out["source"].to_list()))

    assert "g" not in source_of  # replaced
    assert source_of["g_b"] == "dummy"
    assert source_of["h"] == "original"


def test_summary_of_empty_schema_has_headers_but_no_rows() -> None:
    out = nw.from_native(Schema.from_frame(_frame_of("polars", {})).summary())
    assert out.columns == ["name", "dtype", "roles", "source", "sparse"]
    assert out.shape[0] == 0


def test_columns_from_source() -> None:
    schema, _ = _apply(
        [StepDummy(["g"])], "polars", {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}
    )
    assert schema.columns_from_source("dummy") == ["g_b"]
    assert schema.columns_from_source("original") == ["h"]
    assert schema.columns_from_source("nope") == []


def test_parents_of_produced_and_original_columns() -> None:
    schema, _ = _apply(
        [StepDummy(["g"])], "polars", {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}
    )
    assert schema.parents("g_b") == ["g"]
    assert schema.parents("h") == []
    with pytest.raises(KeyError):
        schema.parents("missing")


def test_ancestors_trace_through_archived_columns() -> None:
    # dummy(g) -> g_b, then reduce(g_b, x) -> pc1, pc2: pc1's lineage must reach
    # back through the archived g_b to the original g.
    schema, _ = _apply(
        [StepDummy(["g"]), _FakeReduce(["g_b", "x"])],
        "polars",
        {"g": ["a", "b", "a"], "x": [1, 2, 3]},
    )
    assert schema.ancestors("pc1") == {"g_b", "x", "g"}
    assert schema.columns_from_source("reduce") == ["pc1", "pc2"]
