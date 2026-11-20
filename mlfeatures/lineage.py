"""The column-lineage DAG.

A recipe records, for every column, where it came from and how its dtype
changed along the way. The record is a **column-lineage DAG** rather than a
flat table: each column is a node, and an edge points from a produced column to
the columns it was built from. It is updated after each step's ``fit()``, so a
fitted recipe can answer provenance questions (which step produced this
column, which original columns does it descend from) and report the live
lineage through ``summary()``.

This module defines the structure (`ColumnNode`, `Lineage`), its construction
from a data frame, the update applied after each step, and the provenance queries
(``summary``, ``columns_from_source``, ``parents``, ``ancestors``).

The DAG is **recipe-side state**: it is not handed to steps (steps receive a
`mlfeatures.types.SchemaInfo`, which only needs roles). It is consumed by
``recipe.summary()``, the provenance queries, and the ``has_source`` and
``has_ancestor`` selectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures.types import DataFrame, Frame, Roles

if TYPE_CHECKING:
    from mlfeatures.step import Step

__all__ = ["ColumnNode", "Lineage"]


@dataclass
class ColumnNode:
    """One column's lineage record.

    A node is mutable: an in-place step (e.g. `StepNormalize`) updates
    ``dtype`` and appends to ``dtype_history`` without creating a new node, and
    a step that drops a column sets ``removed_by`` and the node is archived
    rather than discarded, so ancestry still resolves through it. Live nodes
    are read with ``lineage["name"]``; archived ones through `Lineage.removed`.

    Attributes
    ----------
    name
        Current column name.
    dtype
        Current narwhals dtype (may change across steps).
    roles
        Current role names, as a set (unordered).
    source
        Which step *produced* this column: ``"original"`` for an input column,
        otherwise the producing step's ``id``. An in-place transform does not
        change ``source``: it does not *produce* a column.
    parent_cols
        The input column name(s) this column descends from. Empty for original
        columns and for parent-less produced columns (e.g. those added by
        `StepIntercept`).
    dtype_history
        The dtype trajectory: the creation dtype followed by an entry for each
        step that *changed* the dtype (not one per step boundary).
    sparse
        Whether this column is currently sparse.
    removed_by
        ``None`` while the column is live; set to the dropping step's ``id``
        once it has been removed from the data frame and archived.

    Examples
    --------
    Nodes are read off a fitted recipe's `Lineage` by column name. Here a
    numeric column is normalized in place, and `StepDummy` then expands a
    categorical column:

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, StepDummy
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit
    ```

    ```{python}
    recipe = (
        Recipe(target="Status")
        .add(StepNormalize(s.numeric_features()))
        .add(StepDummy(s.nominal_features()))
    )
    fitted = recipe.fit(credit)
    lineage = fitted.lineage
    node = lineage["Home_rent"]
    node.name, node.source, node.parent_cols
    ```

    ``Age`` was normalized in place, so it is still an ``"original"`` column
    but its dtype history records the integer to float cast:

    ```{python}
    lineage["Age"].source, lineage["Age"].dtype_history
    ```

    ``Home`` was consumed by the dummy step, so it is archived with
    ``removed_by`` set:

    ```{python}
    [(n.name, n.removed_by) for n in lineage.removed if n.name == "Home"]
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


class Lineage:
    """The record of where every column came from, and what happened to it.

    A recipe can change a data frame a lot: it adds columns (dummy indicators,
    polynomial terms), removes others (the categorical a dummy step consumed),
    and changes dtypes in place (an integer column becomes float after
    normalizing). After a few steps the output no longer looks much like the
    input, and column names alone do not say why a column exists. A ``Lineage``
    keeps that history so a fitted recipe can answer questions such as "which
    step produced ``Home_rent``?", "which original columns does it depend on?"
    and "what dtype did ``Age`` have before it was normalized?". The same
    record backs ``summary()`` on a fitted recipe and the ``has_source`` and
    ``has_ancestor`` selectors, which pick columns by where they came from
    rather than by name.

    It is stored as a graph with one node per column (a `ColumnNode`). Each
    produced column points at its **parents**, the columns it was built from, so
    following those links back through the steps gives its **ancestors**: every
    column it ultimately descends from. A column that was in the input data has
    no parents and a ``source`` of ``"original"``; a column added by a step has
    the step's ``id`` as its ``source``.

    Columns come in two kinds:

    - **Live** columns are the ones currently in the data frame, in column
      order. These are what ``lineage["name"]``, ``"name" in lineage``,
      ``len(lineage)``, ``columns`` and ``dtypes`` describe.
    - **Removed** columns were dropped by a step. They are kept (see
      ``removed``) instead of being forgotten, because a live column's history
      often runs *through* one. A dummy indicator's parent is the categorical it
      was encoded from; if that categorical were forgotten when the dummy step
      dropped it, the indicator could no longer be traced to the original data.
      ``parents`` and ``ancestors`` therefore resolve removed columns too, and
      asking about a column that never existed raises ``KeyError``.

    The recipe builds the lineage and updates it after each step, so it is read
    rather than constructed: get it from ``FittedRecipe.lineage``.

    Examples
    --------
    Here a small pipeline imputes missing values, rescales the numerics, and
    dummy-encodes the categoricals, composed step by step with ``.add()``:

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepImputeMean, StepNormalize, StepDummy
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit
    ```

    ```{python}
    recipe = (
        Recipe(target="Status")
        .add(StepImputeMean(s.numeric_features()))
        .add(StepNormalize(s.numeric_features()))
        .add(StepDummy(s.nominal_features()))
    )
    fitted = recipe.fit(credit)
    lineage = fitted.lineage
    lineage.summary()
    ```

    ``Age`` stayed in place through the numeric steps, so its ``source`` is
    still ``"original"``, but ``dtype_history`` records the integer to float
    cast that `StepNormalize` applied:

    ```{python}
    lineage["Age"].source, lineage["Age"].dtype_history
    ```

    ``Home`` was expanded into indicators and dropped. ``columns_from_source``
    grabs everything a step produced:

    ```{python}
    lineage.columns_from_source("dummy")
    ```

    ``ancestors`` traces an indicator back through the now-removed ``Home``
    column:

    ```{python}
    lineage.ancestors("Home_rent")
    ```
    """

    # Node for each column currently in the data frame, keyed by name, in column
    # order.
    _live: dict[str, ColumnNode]
    # Nodes of removed columns, in removal order.
    _history: list[ColumnNode]

    def __init__(self) -> None:
        self._live = {}
        self._history = []

    @classmethod
    def from_frame(cls, frame: Frame, roles: Roles | None = None) -> "Lineage":
        """Build the initial DAG: one live node per column, all ``"original"``.

        Each node takes its dtype from the data frame, its roles from ``roles`` (an
        empty `Roles` if none given), an empty ``parent_cols``, a
        single-element ``dtype_history``, and ``sparse=False``.

        Examples
        --------
        ```{python}
        import pandas as pd
        from mlfeatures import _frame
        from mlfeatures.lineage import Lineage

        frame = _frame.from_native(
            pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        )
        lineage = Lineage.from_frame(frame)
        lineage.columns
        ```

        Every node starts as an ``"original"`` with no parents:

        ```{python}
        lineage["a"].source, lineage["a"].parent_cols
        ```
        """
        roles = roles if roles is not None else Roles()
        lineage = cls()
        for name, dtype in _frame.dtypes(frame).items():
            lineage._live[name] = ColumnNode(
                name=name,
                dtype=dtype,
                roles=set(roles.of(name)),
                source="original",
                parent_cols=[],
                dtype_history=[dtype],
                sparse=False,
            )
        return lineage

    def check_additions(self, step_id: str, produced: list[str]) -> None:
        """Validate the columns a step will *add*, before it runs.

        ``produced`` is the step's ``produced_columns()``: an ordered list of
        the names it adds, preserving duplicates. This is the single centralized
        name-collision check: it rejects a produced name that already exists in
        the live lineage *and* a name the step would emit more than once. Running
        before ``step.transform()`` keeps the error clean and identical on every
        backend; left to transform time, polars raises a backend-specific error
        and pandas silently keeps only one of the colliding columns.

        ``update()`` re-checks collisions against the *post-transform* data frame as a
        backstop, but cannot see a within-step duplicate (the data frame has already
        collapsed the two columns into one), which is why this guard runs here.
        """
        live = set(self._live)
        seen: set[str] = set()
        existing: list[str] = []
        duplicated: list[str] = []
        for name in produced:
            if name in live and name not in existing:
                existing.append(name)
            elif name in seen and name not in duplicated:
                duplicated.append(name)
            seen.add(name)
        if existing:
            raise ValueError(
                f"step {step_id!r} produces column(s) {sorted(existing)} that "
                "already exist in the lineage; output column names must be unique"
            )
        if duplicated:
            raise ValueError(
                f"step {step_id!r} produces duplicate column name(s) "
                f"{sorted(duplicated)}; output column names must be unique"
            )

    def update(self, step: "Step", frame_after: Frame, roles: Roles) -> None:
        """Fold one fitted, transformed step into the DAG.

        Called by the recipe after ``step.fit()`` and (for non-``RowOp`` steps)
        ``step.transform()``. The algorithm is **driven by the data frame**: the
        transformed data frame is the source of truth for which columns now exist, their order,
        and their dtypes, while ``step.output_parents()`` is the source of truth
        for the provenance of any *new* columns. ``step.mode`` is not consulted;
        the shape is read off the data frame.

        Concretely, comparing the live columns against ``frame_after``:

        - **Removed** columns (live, now gone) are archived into ``_history``
          with ``removed_by`` set, so ancestry still resolves through them.
        - **Surviving** columns have their roles refreshed and, if the dtype
          changed, the new dtype appended to ``dtype_history``. ``source`` is
          left unchanged: an in-place transform does not *produce* a column.
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
        them), so ``frame_after`` always reflects a real lineage change or none.
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
                "already exist in the lineage; output column names must be unique"
            )
        if set(parents) != set(added):
            raise ValueError(
                f"step {step.id!r}: declared produced columns {sorted(parents)} "
                f"do not match the columns it added to the data frame {sorted(added)}"
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
                f"{sorted(unknown_parents)} that are not in the lineage"
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

        # 4. reorder the live view to match the data frame
        self._live = {name: self._live[name] for name in after}

    def refresh_roles(self, roles: Roles) -> None:
        """Re-read every live column's roles from ``roles``.

        Called by the recipe after a role operation, which changes roles
        without changing any column, so `update` is not run.
        """
        for name, node in self._live.items():
            node.roles = set(roles.of(name))

    @property
    def columns(self) -> list[str]:
        """Live column names, in data frame order."""
        return list(self._live)

    @property
    def dtypes(self) -> dict[str, nw.dtypes.DType]:
        """Map of live column name to its current narwhals dtype."""
        return {name: node.dtype for name, node in self._live.items()}

    @property
    def removed(self) -> list[ColumnNode]:
        """Archived nodes of the columns that steps have removed.

        In removal order. Each node has ``removed_by`` set to the id of the step
        that dropped it. These are the intermediates that `parents` and
        `ancestors` trace through; a column dropped and later re-created under
        the same name appears here once per removal.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit
        ```

        ```{python}
        recipe = Recipe(target="Status").add(StepDummy(s.nominal_features()))
        fitted = recipe.fit(credit)
        lineage = fitted.lineage
        ```

        Each removed node records which step dropped it:

        ```{python}
        [(node.name, node.removed_by) for node in lineage.removed]
        ```
        """
        return list(self._history)

    def summary(self, include_removed: bool = False) -> DataFrame:
        """The lineage as a tidy data frame.

        Columns ``name``, ``dtype``, ``roles``, ``source``, ``sparse``: one row
        per live column, in data frame order. ``dtype`` is the dtype category
        (``"float"``, ``"int"``, ``"str"``, ``"categorical_unordered"``, ...) and
        ``roles`` is the sorted role set joined with ``", "``.

        The result is always a polars ``DataFrame``, whatever backend the
        recipe was fitted on.

        This is the same table `FittedRecipe.summary` returns for the live
        columns. Reach for it when you also want the removed columns
        (``include_removed=True``) and which step dropped each; use `parents`
        and `ancestors` to trace a single column.

        Parameters
        ----------
        include_removed
            If ``True``, append a row for each removed column (in removal
            order) and add two columns: ``parents``, the immediate parent
            columns joined with ``", "``, and ``removed_by``, the id of the step
            that dropped the column (null for live columns). Default ``False``.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        recipe = (
            Recipe(target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        fitted = recipe.fit(credit)
        lineage = fitted.lineage
        ```

        One row per live column, mixing the normalized numerics (source
        ``"original"``, now ``float``) with the dummy indicators (source
        ``"dummy"``):

        ```{python}
        lineage.summary()
        ```

        With ``include_removed=True`` the dropped categoricals appear too, along
        with each column's parents:

        ```{python}
        lineage.summary(include_removed=True)
        ```
        """
        nodes = list(self._live.values())
        if include_removed:
            nodes += self._history
        data: dict[str, list[Any]] = {
            "name": [n.name for n in nodes],
            "dtype": [_frame.dtype_category(n.dtype) for n in nodes],
            "roles": [", ".join(sorted(n.roles)) for n in nodes],
            "source": [n.source for n in nodes],
            "sparse": [n.sparse for n in nodes],
        }
        if include_removed:
            data["parents"] = [", ".join(n.parent_cols) for n in nodes]
            data["removed_by"] = [n.removed_by for n in nodes]
        return nw.from_dict(data, backend="polars").to_native()

    def columns_from_source(self, step_id: str) -> list[str]:
        """Live columns produced by the step with ``id == step_id``.

        This is what the ``has_source`` selector resolves to. A step's ``id`` is
        the one passed to it, or by default the step type's short name
        (``"dummy"`` for `StepDummy`); a second step of the same type in one
        recipe becomes ``"dummy_2"``.

        Parameters
        ----------
        step_id
            The producing step's ``id``, or ``"original"`` for the input columns
            that survive to the live lineage.

        Returns
        -------
        list of str
            Matching column names in data frame order; empty if the step produced no
            live column or no step has that id.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        recipe = (
            Recipe(target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        fitted = recipe.fit(credit)
        lineage = fitted.lineage
        ```

        Select every column a step produced (the dummy indicators):

        ```{python}
        lineage.columns_from_source("dummy")
        ```

        ``"original"`` picks the inputs that survive to the output (the numerics
        and the target are still here; the categoricals were consumed by the
        dummy step):

        ```{python}
        lineage.columns_from_source("original")
        ```
        """
        return [name for name, node in self._live.items() if node.source == step_id]

    def parents(self, name: str) -> list[str]:
        """Immediate parent columns of ``name`` (live or archived).

        Parameters
        ----------
        name
            A column that is live or has been removed.

        Returns
        -------
        list of str
            The columns ``name`` was built from; empty for an original column.

        Raises
        ------
        KeyError
            If no column called ``name`` ever existed.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        recipe = Recipe(target="Status").add(StepDummy(s.nominal_features()))
        fitted = recipe.fit(credit)
        lineage = fitted.lineage
        ```

        Each indicator's immediate parent is the column it was encoded from:

        ```{python}
        lineage.parents("Home_rent")
        ```
        """
        node = self._node(name)
        if node is None:
            raise KeyError(name)
        return list(node.parent_cols)

    def ancestors(self, name: str) -> set[str]:
        """All columns ``name`` descends from, transitively.

        Walks ``parent_cols`` through archived nodes too, so a column traces
        back through removed intermediates (e.g. a polynomial term back to a
        distance column back to the numeric features it was computed from). The
        result is an unordered set; wrap it in ``sorted()`` for stable display.

        Parameters
        ----------
        name
            A column that is live or has been removed.

        Returns
        -------
        set of str
            Every column ``name`` descends from; empty for an original column.

        Raises
        ------
        KeyError
            If no column called ``name`` ever existed.

        Examples
        --------
        ``ancestors`` walks the whole chain, even through columns that later
        steps removed. Here `StepClassDistance` builds ``distance_good`` from
        the numeric features, then `StepPolynomial` consumes ``distance_good``
        and expands it into polynomial terms:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepImputeMean, StepClassDistance
        from mlfeatures import StepPolynomial
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit
        ```

        ```{python}
        recipe = (
            Recipe(target="Status")
            .add(StepImputeMean(s.numeric_features()))
            .add(StepClassDistance(s.numeric_features(), target="Status"))
            .add(StepPolynomial(s.col("distance_good"), degree=2))
        )
        fitted = recipe.fit(credit)
        lineage = fitted.lineage
        lineage.columns
        ```

        A polynomial term traces back through the now-removed ``distance_good``
        to the original features it was built from:

        ```{python}
        sorted(lineage.ancestors("distance_good_1"))
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
        return f"Lineage(columns={self.columns!r}, removed={len(self._history)})"
