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
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.types import DataFrame, Frame, Roles

if TYPE_CHECKING:
    from shoez.step import Step

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

    Examples
    --------
    Nodes are read off a fitted recipe's `Schema` by column name:

    ```{python}
    import pandas as pd
    from shoez import Recipe, StepDummy

    df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
    node = Recipe(steps=[StepDummy(["color"])]).fit(df).schema["color_red"]
    node.name, node.source, node.parent_cols
    ```
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

    Examples
    --------
    The DAG is built by the recipe; read it back from ``FittedRecipe.schema``.

    ```{python}
    import pandas as pd
    from shoez import Recipe, StepDummy

    df = pd.DataFrame(
        {"color": ["red", "green", "blue", "red"], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    schema = Recipe(steps=[StepDummy(["color"])]).fit(df).schema
    ```

    ``color`` was expanded into indicators and dropped; ``x`` passed through:

    ```{python}
    schema.columns
    ```

    Provenance resolves — each indicator is sourced to the step and traces back
    to ``color`` (gone from the live schema, but kept in the DAG):

    ```{python}
    schema.columns_from_source("dummy")
    ```

    ```{python}
    sorted(schema.ancestors("color_red"))
    ```

    ``summary()`` renders the whole live schema as a tidy table:

    ```{python}
    schema.summary()
    ```
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

        Examples
        --------
        ```{python}
        import pandas as pd
        from shoez import _frame
        from shoez.schema import Schema

        frame = _frame.from_native(
            pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        )
        schema = Schema.from_frame(frame)
        schema.columns
        ```

        Every node starts as an ``"original"`` with no parents:

        ```{python}
        schema["a"].source, schema["a"].parent_cols
        ```
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

    def update(self, step: "Step", frame_after: Frame, roles: Roles) -> None:
        """Fold one fitted, transformed step into the DAG.

        Called by the recipe after ``step.fit()`` and (for non-``RowOp`` steps)
        ``step.transform()``. The algorithm is **frame-driven**: the transformed
        frame is the source of truth for which columns now exist, their order,
        and their dtypes, while ``step.output_parents()`` is the source of truth
        for the provenance of any *new* columns. ``step.mode`` is not consulted
        — the shape is read off the frame.

        Concretely, comparing the live columns against ``frame_after``:

        - **Removed** columns (live, now gone) are archived into ``_history``
          with ``removed_by`` set, so ancestry still resolves through them.
        - **Surviving** columns have their roles refreshed and, if the dtype
          changed, the new dtype appended to ``dtype_history``. ``source`` is
          left unchanged — an in-place transform does not *produce* a column.
        - **Added** columns become new nodes with ``source = step.id``,
          ``parent_cols`` from ``output_parents()``, and ``sparse`` from
          ``produces_sparse``.

        Two integrity guards run first (before any mutation):

        - **Uniqueness:** a produced column name must not collide with a column
          that already exists.
        - **Coverage:** ``output_parents()`` must declare exactly the columns
          that actually appeared, and every declared parent must be a column
          that was present before the step. A mismatch is a step bug.

        ``RowOp`` and empty-selection steps never reach here (the recipe skips
        them), so ``frame_after`` always reflects a real schema change or none.
        """
        after = _frame.columns(frame_after)
        after_set = set(after)
        new_dtypes = _frame.dtypes(frame_after)
        live_before = set(self._live)
        added = [name for name in after if name not in live_before]
        removed = [name for name in self._live if name not in after_set]
        parents = step.output_parents()

        # --- integrity guards (no mutation yet) ---
        collisions = set(parents) & live_before
        if collisions:
            raise ValueError(
                f"step {step.id!r} produces column(s) {sorted(collisions)} that "
                "already exist in the schema; output column names must be unique"
            )
        if set(parents) != set(added):
            raise ValueError(
                f"step {step.id!r}: declared produced columns {sorted(parents)} "
                f"do not match the columns it added to the frame {sorted(added)}"
            )
        unknown_parents = {
            parent
            for parents_of in parents.values()
            for parent in parents_of
            if parent not in live_before
        }
        if unknown_parents:
            raise ValueError(
                f"step {step.id!r} declares parent column(s) "
                f"{sorted(unknown_parents)} that are not in the schema"
            )

        # 1. archive removed columns (retain for provenance)
        for name in removed:
            node = self._live.pop(name)
            node.removed_by = step.id
            self._history.append(node)

        # 2. refresh surviving columns in place (source unchanged)
        for name in after:
            if name in self._live:
                node = self._live[name]
                node.roles = set(roles.of(name))
                if new_dtypes[name] != node.dtype:
                    node.dtype = new_dtypes[name]
                    node.dtype_history.append(new_dtypes[name])

        # 3. add new columns with their provenance
        for name in added:
            self._live[name] = ColumnNode(
                name=name,
                dtype=new_dtypes[name],
                roles=set(roles.of(name)),
                source=step.id,
                parent_cols=list(parents[name]),
                dtype_history=[new_dtypes[name]],
                sparse=step.produces_sparse,
            )

        # 4. reorder the live view to match the frame
        self._live = {name: self._live[name] for name in after}

    @property
    def columns(self) -> list[str]:
        """Live column names, in frame order."""
        return list(self._live)

    @property
    def dtypes(self) -> dict[str, nw.dtypes.DType]:
        """Map of live column name to its current narwhals dtype."""
        return {name: node.dtype for name, node in self._live.items()}

    def summary(self) -> DataFrame:
        """The live schema as a tidy frame (see ``design/schema.md``).

        Columns ``name``, ``dtype``, ``roles``, ``source``, ``sparse`` — one row
        per live column, in frame order. ``dtype`` is the taxonomy category
        (``"float"``, ``"categorical_unordered"``, …) from
        `shoez._frame.dtype_category`; ``roles`` is the sorted role set joined
        with ``", "``. Built on the polars backend and returned native.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepDummy

        df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
        schema = Recipe(steps=[StepDummy(["color"])]).fit(df).schema
        ```

        For a recipe that dummy-encodes ``color`` — one row per live column:

        ```{python}
        schema.summary()
        ```
        """
        nodes = list(self._live.values())
        data: dict[str, list[Any]] = {
            "name": [n.name for n in nodes],
            "dtype": [_frame.dtype_category(n.dtype) for n in nodes],
            "roles": [", ".join(sorted(n.roles)) for n in nodes],
            "source": [n.source for n in nodes],
            "sparse": [n.sparse for n in nodes],
        }
        return nw.from_dict(data, backend="polars").to_native()

    def columns_from_source(self, step_id: str) -> list[str]:
        """Live columns produced by the step with ``id == step_id``.

        The primitive behind the ``has_source(step_id)`` selector (#20). Pass
        ``"original"`` for the input columns that survive to the live schema.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepDummy

        df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
        schema = Recipe(steps=[StepDummy(["color"])]).fit(df).schema
        ```

        ```{python}
        schema.columns_from_source("dummy")
        ```
        """
        return [name for name, node in self._live.items() if node.source == step_id]

    def parents(self, name: str) -> list[str]:
        """Immediate parent columns of ``name`` (live or archived).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepDummy

        df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
        schema = Recipe(steps=[StepDummy(["color"])]).fit(df).schema
        ```

        ```{python}
        schema.parents("color_green")
        ```
        """
        node = self._node(name)
        if node is None:
            raise KeyError(name)
        return list(node.parent_cols)

    def ancestors(self, name: str) -> set[str]:
        """All columns ``name`` descends from, transitively.

        Walks ``parent_cols`` through archived nodes too, so a column traces
        back through removed intermediates (e.g. a reduced component back to a
        dummy column back to the categorical it replaced). Cycles — which a DAG
        should never contain — are guarded against defensively.

        Examples
        --------
        ``color`` was dropped by the dummy step, but the indicator still traces
        back to it through the archived node:

        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepDummy

        df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
        schema = Recipe(steps=[StepDummy(["color"])]).fit(df).schema
        ```

        ```{python}
        sorted(schema.ancestors("color_red"))
        ```
        """
        result: set[str] = set()
        stack = list(self.parents(name))
        while stack:
            parent = stack.pop()
            if parent in result:
                continue
            result.add(parent)
            node = self._node(parent)
            if node is not None:
                stack.extend(node.parent_cols)
        return result

    def _node(self, name: str) -> ColumnNode | None:
        """The node named ``name``, preferring the live one, else most-recently
        archived. ``None`` if no such column ever existed."""
        if name in self._live:
            return self._live[name]
        for node in reversed(self._history):
            if node.name == name:
                return node
        return None

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
