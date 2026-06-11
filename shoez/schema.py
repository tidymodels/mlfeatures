"""The column-lineage DAG.

The recipe maintains an internal **column-lineage DAG** — not a flat table —
that is updated after each step's ``fit()`` (see ``design/schema.md``). R
``recipes`` used a flat ``var_info`` / ``term_info`` table, which made
provenance impossible and caused years of issues. A DAG instead records, per
column, where it came from and how its dtype evolved, so that provenance
queries (``has_source``, ancestry) resolve and ``recipe.summary()`` can report
the live schema.

This module defines the structure (`ColumnNode`, `Schema`) and its initial
construction from a frame. The ``update()`` step-application logic, ``summary()``
and the provenance queries land in later commits of issue #19.

The DAG is **recipe-side state**: it is not handed to steps (steps receive a
`shoez.types.SchemaInfo`, which only needs roles). It is consumed by
``recipe.summary()``, provenance queries, and the ``has_source`` selector.
"""

from __future__ import annotations

from dataclasses import dataclass

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.types import Frame, Roles

__all__ = ["ColumnNode", "Schema"]


@dataclass
class ColumnNode:
    """One column's lineage record. See ``design/schema.md``.

    A node is mutable: an in-place step (e.g. ``step_normalize``) updates
    ``dtype`` and appends to ``dtype_history`` without creating a new node, and
    a step that drops a column sets ``removed_by`` and the node is archived
    rather than discarded — so ancestry still resolves through it.

    Attributes
    ----------
    name
        Current column name.
    dtype
        Current narwhals dtype (may change across steps).
    roles
        Current role labels (see ``design/roles.md``).
    source
        Which step *produced* this column: ``"original"`` for an input column,
        otherwise the producing step's ``id``. An in-place transform does not
        change ``source`` — it does not *produce* a column.
    parent_cols
        The input column name(s) this column descends from. Empty for original
        columns and for parent-less produced columns (e.g. ``step_intercept``).
    dtype_history
        The dtype trajectory: the creation dtype followed by an entry for each
        step that *changed* the dtype (not one per step boundary).
    sparse
        Whether this column is currently sparse (see ``design/ecosystem.md``).
    removed_by
        ``None`` while the column is live; set to the dropping step's ``id``
        once it has been removed from the frame and archived.
    """

    name: str
    dtype: nw.dtypes.DType
    roles: set[str]
    source: str
    parent_cols: list[str]
    dtype_history: list[nw.dtypes.DType]
    sparse: bool = False
    removed_by: str | None = None


class Schema:
    """The column-lineage DAG: live columns plus archived removed columns.

    ``_live`` maps each current column name to its `ColumnNode`, in frame
    order. ``_history`` retains the nodes of columns that have been removed, so
    provenance can be traced *through* a dropped intermediate (e.g. a dummy
    column back to the categorical it replaced). Together they form the DAG;
    ``columns`` / ``dtypes`` and (later) ``summary()`` expose only the live
    view, while the provenance queries also consult history.

    Build the initial DAG with `from_frame`; the recipe then calls ``update()``
    after each step to keep it in sync with the transformed frame.
    """

    def __init__(self) -> None:
        self._live: dict[str, ColumnNode] = {}
        self._history: list[ColumnNode] = []

    @classmethod
    def from_frame(cls, frame: Frame, roles: Roles | None = None) -> "Schema":
        """Build the initial DAG: one live node per column, all ``"original"``.

        Each node takes its dtype from the frame, its roles from ``roles`` (an
        empty `Roles` if none given), an empty ``parent_cols``, a
        single-element ``dtype_history``, and ``sparse=False``.
        """
        roles = roles if roles is not None else Roles()
        schema = cls()
        for name, dtype in _frame.dtypes(frame).items():
            schema._live[name] = ColumnNode(
                name=name,
                dtype=dtype,
                roles=set(roles.of(name)),
                source="original",
                parent_cols=[],
                dtype_history=[dtype],
                sparse=False,
            )
        return schema

    @property
    def columns(self) -> list[str]:
        """Live column names, in frame order."""
        return list(self._live)

    @property
    def dtypes(self) -> dict[str, nw.dtypes.DType]:
        """Map of live column name to its current narwhals dtype."""
        return {name: node.dtype for name, node in self._live.items()}

    def __getitem__(self, name: str) -> ColumnNode:
        """The live node named ``name`` (raises ``KeyError`` if not live)."""
        return self._live[name]

    def __contains__(self, name: object) -> bool:
        """Whether ``name`` is a live column."""
        return name in self._live

    def __len__(self) -> int:
        """Number of live columns."""
        return len(self._live)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Schema(columns={self.columns!r}, removed={len(self._history)})"
