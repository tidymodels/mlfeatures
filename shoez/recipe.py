"""The `Recipe` lifecycle driver.

A minimal, vertical-slice implementation of the ``specify → fit → transform``
lifecycle described in ``design/lifecycle.md``. It exercises the step contract
end to end and maintains the column-lineage DAG (``design/schema.md``) across
``fit()``. At ``fit()`` it snapshots the training schema into a ``Ptype`` and
validates the incoming frame against it; the snapshot rides along on the
``FittedRecipe`` so ``transform()`` can re-check new data at the boundary before
any step runs (see ``design/lifecycle.md``).

Steps are pure: ``Recipe(steps=[...])`` and ``.add(...)`` return a *new*
``Recipe`` with the step appended — they never touch data. ``fit()`` runs each step's ``fit`` on the
training data in order, feeding the transformed output of one step into the next
(``RowOp`` steps are fitted but not applied in the fit loop). ``transform()``
applies the fitted steps to new data, skipping ``RowOp`` steps entirely.
"""

from __future__ import annotations

from shoez import _frame
from shoez.ptype import (
    _NON_CONFIGURABLE_ROLES,  # pyright: ignore[reportPrivateUsage]
    Ptype,
)
from shoez.schema import Schema
from shoez.selectors import ShoezSelector
from shoez.step import RowOp, Step, StepError
from shoez.types import (
    DataFrame,
    Frame,
    Roles,
    SchemaInfo,
    _apply_role_assignments,  # pyright: ignore[reportPrivateUsage]
)

__all__ = ["Recipe", "FittedRecipe"]


def _format_recipe(
    *, prepped: bool, roles: Roles, steps: list[Step], n_columns: int | None
) -> str:
    """Render the shared ``repr`` block for `Recipe` / `FittedRecipe`.

    Follows the layout in ``design/inspection.md``: a header tagging the
    training state, an ``Inputs`` line summarising column roles, and the steps
    aligned so their ``[trained]`` / ``[untrained]`` markers line up.
    """
    header = f"Recipe ({'prepped' if prepped else 'unprepped'})"

    labels = roles.role_labels()
    if labels:
        parts: list[str] = []
        for label in labels:
            count = len(roles.columns_with_role(label))
            name = label if count == 1 or label.endswith("s") else f"{label}s"
            parts.append(f"{count} {name}")
        role_summary = ", ".join(parts)
    else:
        role_summary = "no roles assigned"
    if n_columns is not None:
        inputs = f"  Inputs: {n_columns} columns — {role_summary}"
    else:
        inputs = f"  Inputs: {role_summary}"

    lines = [header, inputs, "  Steps:"]
    if not steps:
        lines.append("    <none>")
        return "\n".join(lines)

    heads = [step._repr_head() for step in steps]  # pyright: ignore[reportPrivateUsage]
    width = max(len(head) for head in heads)
    for step, head in zip(steps, heads):
        marker = "trained" if step.trained else "untrained"
        lines.append(f"    {head.ljust(width)}  [{marker}]")
    return "\n".join(lines)


class Recipe:
    """A declarative, unfitted pipeline of feature-engineering steps.

    Steps are first-class `Step` objects (Option A in
    ``design/interfaces.md``). Assemble a recipe by passing them at construction:

    ```python
    Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)])
    ```

    or with the immutable `add()` builder, which returns a new recipe each
    call (preserving the purity guarantee in ``design/lifecycle.md``):

    ```python
    Recipe().add(StepDropCorr(["a", "b", "c"], threshold=0.9))
    ```

    A prototype ``data`` frame may be supplied to capture the column names used
    when assigning roles. Roles are declared via ``outcome``, ``predictors``,
    and ``case_weights`` keyword arguments.
    """

    def __init__(
        self,
        steps: list[Step] | None = None,
        *,
        data: DataFrame | None = None,
        outcome: str | list[str] | None = None,
        predictors: str | list[str] | None = None,
        case_weights: str | list[str] | None = None,
    ):
        self.steps: list[Step] = list(steps) if steps is not None else []
        # The optional prototype frame backs column-existence checks during role
        # assignment and the pre-fit ``summary()``. It is never mutated, so
        # clones share the reference. The lineage DAG is derived from it on
        # demand rather than stored — see ``summary()``.
        self._prototype_frame: Frame | None = (
            _frame.from_native(data) if data is not None else None
        )
        self.roles = Roles()
        #: Pending transform-time requirement overrides, applied at ``fit()``.
        #: Each entry is a (target, transform_required) pair where target is a
        #: role name or a selector resolved against the training frame.
        self._role_req_overrides: list[tuple[str | ShoezSelector, bool]] = []
        _apply_role_assignments(
            self.roles,
            {"outcome": outcome, "predictor": predictors, "case_weights": case_weights},
            check_col=self._check_column_exists,
        )

    def __repr__(self) -> str:
        """A human-readable, unprepped view of the pipeline (``design/inspection.md``).

        Shows the declared role assignments and the steps with their unresolved
        selectors, each tagged ``[untrained]``. Column counts come from the
        prototype frame when one was supplied; otherwise only the explicitly
        assigned roles are summarised. ``FittedRecipe`` has its own ``repr`` that
        shows resolved columns and the ``prepped`` state instead.
        """
        n_columns = (
            len(_frame.columns(self._prototype_frame))
            if self._prototype_frame is not None
            else None
        )
        return _format_recipe(
            prepped=False, roles=self.roles, steps=self.steps, n_columns=n_columns
        )

    def _clone(self) -> "Recipe":
        """A shallow copy that preserves steps, prototype, roles, and overrides."""
        new = Recipe(steps=self.steps)
        new._prototype_frame = self._prototype_frame
        new.roles = self.roles.copy()
        new._role_req_overrides = list(self._role_req_overrides)
        return new

    def add(self, *steps: Step) -> "Recipe":
        """Return a new recipe with ``steps`` appended (steps are pure).

        Examples
        --------
        Each ``add()`` returns a new recipe, so steps chain into a pipeline
        without mutating the original:

        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepNormalize, StepDummy
        ```

        ```{python}
        Recipe().add(StepNormalize(["age"])).add(StepDummy(["city"]))
        ```
        """
        new = self._clone()
        new.steps = [*self.steps, *steps]
        return new

    def set_roles(
        self,
        *,
        outcome: str | list[str] | None = None,
        predictor: str | list[str] | None = None,
        case_weights: str | list[str] | None = None,
        **custom: str | list[str],
    ) -> "Recipe":
        """Return a new recipe with the given role assignments added.

        Standard roles have dedicated keyword arguments; arbitrary custom roles
        may be passed as additional keywords. Each value is a column name or a
        list of names. Incompatible combinations (e.g. outcome + predictor on
        one column) are rejected.
        """
        raise NotImplementedError("set_roles is not yet implemented")

    def update_role_requirements(
        self,
        target: str | ShoezSelector,
        *,
        transform_required: bool,
    ) -> "Recipe":
        """Return a new recipe overriding whether columns are required at transform.

        ``target`` is either a **role name** (a ``str``, e.g. ``"id"``) or a
        selector (e.g. ``starts_with("id_")``) resolved against the training
        frame at ``fit()``. The ``predictor`` and ``outcome`` roles are hardcoded
        and cannot be configured (predictors are always required at transform,
        outcomes never are); passing either as a string ``target`` raises. A
        selector that happens to catch a predictor or outcome column is silently
        a no-op for that column (the hardcoded role wins), so it does not raise.

        See ``design/lifecycle.md`` for the per-role requirement defaults.
        """
        if isinstance(target, str) and target in _NON_CONFIGURABLE_ROLES:
            raise ValueError(f"requirement for role {target!r} is not configurable")
        new = self._clone()
        new._role_req_overrides = [
            *self._role_req_overrides,
            (target, transform_required),
        ]
        return new

    def _resolve_role_req_overrides(
        self, frame: Frame, roles: Roles
    ) -> dict[str, bool]:
        """Resolve pending overrides to a column→requirement map (last wins)."""
        resolved: dict[str, bool] = {}
        for target, required in self._role_req_overrides:
            if isinstance(target, str):
                columns = roles.columns_with_role(target)
            else:
                columns = target.resolve(frame, roles)
            for column in columns:
                resolved[column] = required
        return resolved

    def _check_column_exists(self, column: str) -> None:
        """Validate ``column`` against the prototype, if one was supplied."""
        if self._prototype_frame is not None and column not in _frame.columns(
            self._prototype_frame
        ):
            raise ValueError(f"column {column!r} is not in the prototype data")

    def fit(self, data: DataFrame) -> "FittedRecipe":
        """Fit each step in order on the training data.

        The transformed output of step *i* is the input to step *i+1*'s
        ``fit()``, so later steps see the schema as modified by earlier ones.
        ``RowOp`` steps *are* applied here — a row operation reshapes the
        training frame every later step then fits on — but never run during
        ``transform()`` on new data, which the framework enforces structurally.

        Examples
        --------
        ``fit()`` estimates each step's statistics on the training data and
        returns a `FittedRecipe` with the selectors resolved to concrete
        columns:

        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepNormalize, StepDummy
        ```

        ```{python}
        df = pd.DataFrame(
            {"age": [25, 41, 33, 52], "city": ["NYC", "LA", "NYC", "SF"]}
        )
        Recipe().add(StepNormalize(["age"])).add(StepDummy(["city"])).fit(df)
        ```
        """
        current = _frame.from_native(data)
        live_roles = self.roles.copy()
        if not live_roles.columns_with_role("predictor"):
            reserved = {
                col
                for col in _frame.columns(current)
                if "outcome" in live_roles.of(col)
                or "case_weights" in live_roles.of(col)
            }
            for col in _frame.columns(current):
                if col not in reserved:
                    live_roles.add(col, "predictor")
        # Snapshot the training schema into a ptype and validate the incoming
        # frame against it (all columns required at fit). The snapshot rides on
        # the FittedRecipe so transform() can re-check new data at the boundary.
        overrides = self._resolve_role_req_overrides(current, live_roles)
        ptype = Ptype.from_frame(current, live_roles, overrides)
        ptype.check(current, context="fit")
        # The column-lineage DAG, seeded from the training frame and updated
        # after each column-changing step (see design/schema.md). RowOp and
        # empty-selection steps make no schema change and are skipped below.
        schema = Schema.from_frame(current, live_roles)
        fitted_steps: list[Step] = []
        for step in self.steps:
            info = SchemaInfo.from_frame(current)
            info.roles = live_roles
            # Hand the live lineage DAG and the id→step map to selector
            # resolution so provenance selectors (has_source/has_ancestor) can
            # resolve; these are recipe-side state, not for steps' own logic.
            info.schema = schema
            info.producing_steps = {s.id: s for s in fitted_steps}
            # Empty selection ⇒ identity. Resolved here, once, at the recipe
            # boundary so the no-op is structural: a step is never fitted on a
            # zero-column selection and so cannot misread it (the failure mode
            # the per-step guards used to handle). Steps with no input selector
            # (``StepIntercept``, the ``ADDING`` prototype; ``StepFilter``, a
            # ``RowOp``; ``StepMutate``, the ``UNCONSTRAINED`` escape hatch) are
            # exempt — they have no ``columns`` to resolve and must always run.
            # They are detected structurally by the absence of a ``columns``
            # attribute rather than by enumerating their types. The marker is
            # reset each fit so re-fitting with a selection that now matches
            # columns is not stuck in identity mode.
            has_selector = hasattr(step, "columns")
            selected = step.resolve_selection(current, info) if has_selector else []
            empty = has_selector and selected == []
            step._empty_selection = empty  # pyright: ignore[reportPrivateUsage]
            if empty:
                # Kept in the pipeline (inspectable) but marked identity: skip
                # fit / transform / update_roles.
                step.trained = True
                fitted_steps.append(step)
                continue
            # Only the step's own ``fit``/``transform`` calls are wrapped in a
            # StepError; the recipe-side name-collision guard and schema/role
            # bookkeeping raise their own errors and are left unwrapped.
            try:
                fitted = step.fit(current, info)
            except StepError:
                raise
            except Exception as exc:
                raise StepError.wrap(step, selected, "fit") from exc
            fitted.trained = True  # set by the recipe, not the step (per contract)
            if not isinstance(step, RowOp):
                # Centralized name-collision guard: reject produced columns that
                # clash with the live schema or with each other, before the step
                # runs (see Schema.check_additions). Steps no longer self-check.
                schema.check_additions(fitted.id, fitted.produced_columns())
            # ``transform`` runs at fit time even for ``RowOp`` steps: a row
            # operation only ever takes effect on the *training* frame, so
            # applying it here filters/reshapes the rows every later step fits
            # on. It is skipped only at ``transform()`` time on new data, which
            # the framework enforces structurally by dispatching on ``RowOp``.
            try:
                current = fitted.transform(current)
            except Exception as exc:
                raise StepError.wrap(fitted, selected, "fit") from exc
            fitted.update_roles(live_roles)
            if not isinstance(step, RowOp):
                schema.update(fitted, current, live_roles)
            fitted_steps.append(fitted)
        return FittedRecipe(
            steps=fitted_steps, roles=live_roles, schema=schema, ptype=ptype
        )

    def summary(self) -> DataFrame:
        """The initial (pre-fit) schema as a tidy frame. See ``design/schema.md``.

        Shows the prototype columns before any step runs. Requires a prototype:
        construct the recipe with ``Recipe(data=...)``, or call ``fit()`` and use
        ``FittedRecipe.summary()`` for the post-fit schema.

        Examples
        --------
        The pre-fit summary describes the prototype columns and their roles
        before any step has run, so ``city`` is still a single ``str`` column
        with source ``"original"``:

        ```{python}
        import pandas as pd
        from shoez import Recipe, StepNormalize, StepDummy

        df = pd.DataFrame(
            {
                "price": [10.0, 22.5, 7.0, 30.0],
                "age": [25, 41, 33, 52],
                "city": ["NYC", "LA", "NYC", "SF"],
            }
        )
        rec = (
            Recipe(data=df, outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
        )
        rec.summary()
        ```
        """
        if self._prototype_frame is None:
            raise ValueError(
                "summary() needs prototype data; construct the recipe with "
                "Recipe(data=...), or fit it and call FittedRecipe.summary()"
            )
        return Schema.from_frame(self._prototype_frame, self.roles).summary()


class FittedRecipe:
    """A fitted pipeline: applies stored statistics to new data."""

    def __init__(
        self,
        steps: list[Step],
        roles: Roles | None = None,
        schema: Schema | None = None,
        ptype: Ptype | None = None,
    ):
        self.steps = steps
        self.roles = roles if roles is not None else Roles()
        #: The post-all-steps column-lineage DAG (``design/schema.md``), or
        #: ``None`` if the recipe was constructed without one.
        self.schema = schema
        #: The training-data ptype snapshot (``design/lifecycle.md``) used to
        #: validate new data in ``transform()``, or ``None`` if absent.
        self.ptype = ptype

    def __repr__(self) -> str:
        """A human-readable, prepped view of the fitted pipeline (``design/inspection.md``).

        Shows the post-fit role assignments and each step with the concrete
        columns its selector resolved to at fit time, tagged ``[trained]``. This
        is the resolved counterpart of ``Recipe``'s unresolved ``repr``.
        """
        n_columns = len(self.roles.all_columns()) if self.roles.all_columns() else None
        return _format_recipe(
            prepped=True, roles=self.roles, steps=self.steps, n_columns=n_columns
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a diffable, JSON-safe ``dict``. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.to_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a diffable JSON string. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.to_json(self, indent=indent)

    def to_yaml(self) -> str:
        """Serialize to a diffable YAML string. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.to_yaml(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `to_dict` output. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.from_dict(data)

    @classmethod
    def from_json(cls, text: str) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `to_json` output. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.from_json(text)

    @classmethod
    def from_yaml(cls, text: str) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `to_yaml` output. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.from_yaml(text)

    def minimize(self) -> "FittedRecipe":
        """Return a copy with fit-only artifacts stripped. See `shoez.serialization`."""
        from shoez import serialization

        return serialization.minimize(self)

    def fit(self, data: DataFrame) -> "FittedRecipe":
        """Re-fitting an already-fitted recipe is a no-op: returns ``self``.

        The stored statistics are never re-estimated, so ``rec.fit().fit()`` is
        cheap and yields the same object. ``data`` is ignored.
        """
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        """Apply each fitted step in order, skipping ``RowOp`` and identity steps.

        ``RowOp`` steps never run on new data; steps whose primary selection
        resolved to zero columns at fit time (``_empty_selection``) are the
        identity transform and pass the frame through unchanged. Both are
        skipped here — structurally, by inspecting the step rather than by any
        per-step convention. Never re-estimates. Returns a native frame at the
        sklearn boundary.

        Examples
        --------
        ``transform()`` applies the fitted statistics to new data, returning a
        native frame with the same schema the recipe learned at fit time:

        ```{python}
        #| echo: false
        #| output: false
        import pandas as pd
        from shoez import Recipe, StepNormalize, StepDummy
        ```

        ```{python}
        df = pd.DataFrame(
            {"age": [25, 41, 33, 52], "city": ["NYC", "LA", "NYC", "SF"]}
        )
        fitted = (
            Recipe().add(StepNormalize(["age"])).add(StepDummy(["city"])).fit(df)
        )
        fitted.transform(df)
        ```
        """
        current = _frame.from_native(data)
        if self.ptype is not None:
            self.ptype.check(current, context="transform")
        for step in self.steps:
            if isinstance(step, RowOp) or step._empty_selection:  # pyright: ignore[reportPrivateUsage]
                continue
            try:
                current = step.transform(current)
            except StepError:
                raise
            except Exception as exc:
                columns = step.schema_in().columns if step.trained else []
                raise StepError.wrap(step, columns, "transform") from exc
        return _frame.to_native(current)

    def summary(self) -> DataFrame:
        """The fitted (post-all-steps) schema as a tidy frame. See ``design/schema.md``.

        Examples
        --------
        After fitting, the summary reflects the transformed schema: ``city`` has
        been expanded into dummy indicators (source ``"dummy"``) while the
        normalized ``age`` keeps its predictor role:

        ```{python}
        import pandas as pd
        from shoez import Recipe, StepNormalize, StepDummy

        df = pd.DataFrame(
            {
                "price": [10.0, 22.5, 7.0, 30.0],
                "age": [25, 41, 33, 52],
                "city": ["NYC", "LA", "NYC", "SF"],
            }
        )
        fitted = (
            Recipe(data=df, outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
            .fit(df)
        )
        fitted.summary()
        ```
        """
        if self.schema is None:
            raise RuntimeError("this FittedRecipe was constructed without a schema")
        return self.schema.summary()
