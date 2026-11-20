"""The `Recipe` lifecycle driver.

A minimal, vertical-slice implementation of the ``specify → fit → transform``
lifecycle. It exercises the step contract
end to end and maintains the column-lineage DAG across
``fit()``. At ``fit()`` it snapshots the training data into a ``Schema`` and
validates the incoming data frame against it; the snapshot rides along on the
``FittedRecipe`` so ``transform()`` can re-check new data at the boundary before
any step runs.

Steps are pure: ``Recipe(steps=[...])`` and ``.add(...)`` return a *new*
``Recipe`` with the step appended: they never touch data. ``fit()`` runs each step's ``fit`` on the
training data in order, feeding the transformed output of one step into the next
(``RowOp`` steps are fitted but not applied in the fit loop). ``transform()``
applies the fitted steps to new data, skipping ``RowOp`` steps entirely.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

from mlfeatures import _frame
from mlfeatures.lineage import Lineage
from mlfeatures.schema import (
    _NON_CONFIGURABLE_ROLES,  # pyright: ignore[reportPrivateUsage]
    Schema,
    column_transform_required,
)
from mlfeatures.selectors import FeaturesSelector, Selectors
from mlfeatures.step import (
    RowOp,
    Step,
    StepError,
    _step_type_label,  # pyright: ignore[reportPrivateUsage]
)
from mlfeatures.types import (
    DataFrame,
    Frame,
    NwSelector,
    PlExpr,
    Roles,
    SchemaInfo,
    Selector,
    _apply_role_assignments,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["Recipe", "FittedRecipe"]


def _check_accepted_dtypes(step: Step, data: Frame, selected: list[str]) -> None:
    """Reject a step whose selection includes a dtype it does not accept.

    The framework-side half of ``Step.accepts_dtypes()``: it runs at the recipe
    boundary, after the selector has resolved and before ``step.fit()``, so a
    step never sees a column it declared itself invalid for and a mismatch is
    reported by mlfeatures rather than by whatever narwhals, polars, or pandas
    happens to raise from inside the step. A step returning ``ANY_DTYPE`` opts
    out (the sentinel contains every dtype).
    """
    accepted = step.accepts_dtypes()
    schema = data.schema
    offenders = {
        name: schema[name] for name in selected if type(schema[name]) not in accepted
    }
    if offenders:
        columns = sorted(offenders)
        dtypes = [str(offenders[name]) for name in columns]
        raise TypeError(
            f"step {step.id!r} ({_step_type_label(step)}) does not accept "
            f"column(s) {columns} with dtype(s) {dtypes}"
        )


#: Capitalized names a backend's error message may open with. They read like an
#: ordinary sentence start, so the lowercase-the-first-word rule would mangle them.
_PROPER_NAMES = frozenset({"Polars", "Pandas", "Narwhals", "Modin", "Dask"})


def _lower(exc: Exception) -> str:
    """The text of ``exc`` as one phrase in the house message style.

    Every non-empty line is kept (narwhals puts its "did you mean" suggestion
    on a later line, behind a ``Hint:`` label that reads oddly mid-message and
    is dropped), joined with ``;`` and stripped of a trailing period (an
    ellipsis is kept). A line's first word is lowercased unless it is an
    identifier that is naturally capitalized, such as a class name or the name
    of a backend library.
    """
    parts: list[str] = []
    for line in str(exc).splitlines():
        text = line.strip()
        if text.endswith(".") and not text.endswith("..."):
            text = text.rstrip(".")
        if text[:5].lower() == "hint:":
            text = text[5:].strip()
        if not text:
            continue
        word = text.split(maxsplit=1)[0]
        if (
            len(word) > 1
            and word[1:].islower()
            and word.rstrip(":,;") not in _PROPER_NAMES
        ):
            text = text[0].lower() + text[1:]
        parts.append(text)
    return "; ".join(parts)


@dataclass(frozen=True)
class _RoleOp:
    """A role operation recorded at a position in the chain.

    ``position`` is the number of steps added before the operation, so it runs
    after that many steps have been fitted and before the next one.
    """

    position: int
    kind: Literal["add", "update", "remove"]
    columns: FeaturesSelector
    role: str  # the role added, the new role, or the role removed
    old_role: str | None = None
    #: The column names, when the selection was given as a plain name or list
    #: of names (recorded from the caller's argument, not read back from the
    #: wrapped selector).
    names: tuple[str, ...] | None = None
    #: Whether the operation was written as ``set_roles()``, which only matters
    #: for how it is shown: it acts like ``add_role()`` at position 0.
    declared: bool = False

    def plain_names(self) -> list[str] | None:
        """The column names, if the selection is a plain name or list of names."""
        return None if self.names is None else list(self.names)

    def describe(self) -> str:
        if self.declared:
            return f"set_roles({self.role}={self.columns!r})"
        if self.kind == "add":
            return f"add_role({self.columns!r}, role={self.role!r})"
        if self.kind == "remove":
            return f"remove_role({self.columns!r}, role={self.role!r})"
        old = "" if self.old_role is None else f", old_role={self.old_role!r}"
        return f"update_role({self.columns!r}, new_role={self.role!r}{old})"

    def apply(
        self,
        frame: Frame,
        roles: Roles,
        lineage: Lineage | None,
        producing_steps: dict[str, Step],
        implicit: set[str] | None = None,
    ) -> list[str]:
        """Resolve the columns against the live data and change ``roles``.

        A selection matching nothing leaves the roles unchanged, like a step.
        ``implicit`` holds the columns whose feature role came from defaulting
        rather than from the user (see `_apply_leading_ops`); it is updated in
        place. Returns the columns the operation resolved to.
        """
        try:
            columns = self.columns.resolve(frame, roles, lineage, producing_steps)
        except Exception as exc:
            raise ValueError(f"{self.describe()} failed; {_lower(exc)}") from exc
        removing = self.old_role if self.kind == "update" else self.role
        if self.kind in ("update", "remove") and removing is not None:
            missing = [c for c in columns if removing not in roles.of(c)]
            if missing:
                raise ValueError(
                    f"{self.describe()} needs column(s) {missing} to carry role "
                    f"{removing!r}"
                )
        # Try on copies first so a failure part-way leaves everything untouched.
        trial = set(implicit) if implicit is not None else None
        try:
            self._change(roles.copy(), columns, trial)
        except ValueError as exc:
            raise ValueError(f"{self.describe()} failed; {_lower(exc)}") from exc
        self._change(roles, columns, implicit)
        return list(columns)

    def _change(
        self, roles: Roles, columns: list[str], implicit: set[str] | None
    ) -> None:
        implicit = set() if implicit is None else implicit
        for column in columns:
            if self.kind == "add":
                self._give(roles, column, implicit)
            elif self.kind == "remove":
                roles.remove(column, self.role)
                if self.role == "feature":
                    implicit.discard(column)
            else:
                previous = (
                    {self.old_role} if self.old_role is not None else roles.of(column)
                )
                for old in previous:
                    roles.remove(column, old)
                if "feature" in previous:
                    implicit.discard(column)
                self._give(roles, column, implicit)

    def _give(self, roles: Roles, column: str, implicit: set[str]) -> None:
        """Add ``self.role`` to ``column``, settling it with the default features.

        Only a feature role that defaulting handed out is ever displaced; one
        the user declared is left to the usual compatibility rules. A
        ``set_roles(feature=...)`` declaration chooses the features, so no other
        default survives it; ``add_role()`` and ``update_role()`` do not, because
        the same call written after a step leaves the other features alone and
        the position of the call must not change what it does.
        """
        if self.role in ("target", "case_weights") and column in implicit:
            # Defaulting never gives these columns the feature role.
            roles.remove(column, "feature")
            implicit.discard(column)
        elif self.role == "feature":
            if self.declared:
                # Naming features is choosing them: the default no longer applies.
                for other in implicit:
                    roles.remove(other, "feature")
                implicit.clear()
            else:
                # The user now holds this feature role on purpose, so a later
                # operation may not quietly displace it as it could a default.
                implicit.discard(column)
        roles.add(column, self.role)


def _default_features(roles: Roles, columns: list[str]) -> set[str]:
    """Give every non-target, non-case-weights column the feature role.

    Only applies when no column carries the role yet. Returns the columns that
    were given the role this way, empty when features were declared, so a role
    operation can tell a default from a choice.
    """
    if roles.columns_with_role("feature"):
        return set()
    given: set[str] = set()
    for col in columns:
        held = roles.of(col)
        if "target" not in held and "case_weights" not in held:
            roles.add(col, "feature")
            given.add(col)
    return given


def _apply_leading_ops(
    ops: list[_RoleOp],
    frame: Frame,
    roles: Roles,
    lineage: Lineage,
    *,
    show_defaults: bool,
) -> set[str]:
    """Run the role operations written before the first step, with defaulting.

    Feature defaulting runs first, so the operations read the same roles they
    would after a step and a role selector such as ``Selectors.numeric_features()``
    sees the default features. What keeps them equal to the ``set_roles()``
    declarations they stand in for is how a default is settled: giving a
    defaulted column the target or case-weights role takes the default feature
    role away (defaulting never hands those columns the role). Adding the
    feature role does not choose the features, as it does not after a step; only
    a ``set_roles(feature=...)`` declaration does, and no other default survives
    that. A feature role the user declared is never displaced.

    ``show_defaults`` is true when fitting. The untrained repr and summary pass
    false and leave out the default features that remain, so they show only
    what was declared.

    Returns the columns whose feature role is still a default, so operations
    written after a step settle defaults the same way these do.
    """
    leading = [op for op in ops if op.position == 0]
    if not leading and not show_defaults:
        return set()
    implicit = _default_features(roles, _frame.columns(frame))
    lineage.refresh_roles(roles)
    for op in leading:
        op.apply(frame, roles, lineage, {}, implicit)
        lineage.refresh_roles(roles)
    if not show_defaults:
        for column in implicit:
            roles.remove(column, "feature")
        return set()
    return implicit


def _format_recipe(
    *,
    trained: bool,
    roles: Roles,
    steps: list[Step],
    n_columns: int | None,
    role_ops: list[_RoleOp] | None = None,
) -> str:
    """Render the shared ``repr`` block for `Recipe` / `FittedRecipe`.

    The layout is a header tagging the
    training state, an ``Inputs`` line summarising column roles, and the steps
    aligned so their ``[trained]`` / ``[untrained]`` markers line up.
    """
    header = f"Recipe ({'trained' if trained else 'untrained'})"

    labels = roles.role_names()
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
        inputs = f"  Inputs: {n_columns} columns ({role_summary})"
    else:
        inputs = f"  Inputs: {role_summary}"

    lines = [header, inputs, "  Steps:"]
    ops = role_ops or []
    if not steps and not ops:
        lines.append("    <none>")
        return "\n".join(lines)

    heads = [step._repr_head() for step in steps]  # pyright: ignore[reportPrivateUsage]
    width = max((len(head) for head in heads), default=0)
    for i in range(len(steps) + 1):
        lines.extend(f"    {op.describe()}" for op in ops if op.position == i)
        if i < len(steps):
            marker = "trained" if steps[i].trained else "untrained"
            lines.append(f"    {heads[i].ljust(width)}  [{marker}]")
    return "\n".join(lines)


def _resolve_step_ids(existing: list[Step], incoming: tuple[Step, ...]) -> list[Step]:
    """Return ``incoming`` with ids made unique against ``existing``.

    Step ids key the lineage DAG, ``SchemaInfo.producing_steps``, the
    ``has_source`` selectors, and step-level error messages, so two steps
    sharing one id turn those lookups into wrong answers rather than errors.
    Uniqueness is enforced here, at the earliest point it is detectable.

    A *defaulted* id (the step type's short name) is disambiguated silently by
    appending the lowest free ``_n`` suffix, so ``StepDummy`` twice in a recipe
    yields ``"dummy"`` and ``"dummy_2"``. Renaming is done on a copy, keeping
    ``add()`` pure with respect to the caller's step objects. An id the caller
    passed explicitly is never rewritten: a duplicate there is a mistake about
    a name the caller intends to use for lookups, so it raises.
    """
    taken = {step.id for step in existing}
    resolved: list[Step] = []
    for step in incoming:
        if step.id not in taken:
            taken.add(step.id)
            resolved.append(step)
            continue
        if step._id_explicit:  # pyright: ignore[reportPrivateUsage]
            raise ValueError(
                f"duplicate step id {step.id!r}; ids must be unique within a recipe"
            )
        n = 2
        while f"{step.id}_{n}" in taken:
            n += 1
        renamed = copy.copy(step)
        renamed.id = f"{step.id}_{n}"
        taken.add(renamed.id)
        resolved.append(renamed)
    return resolved


class Recipe:
    """A declarative, untrained pipeline of feature-engineering steps.

    A recipe is specified in two parts: the columns' roles are declared first
    (the ``target``, ``features`` and ``case_weights`` arguments, plus
    `set_roles()`), then steps are added with `add()`. Specifying a recipe
    never touches data; nothing is estimated until `fit()`, which
    returns a `FittedRecipe`, and `FittedRecipe.transform()` applies it to new
    data.

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepDropCorr, StepNormalize
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit
    ```

    Each `add()` returns a new recipe, so steps chain into a pipeline:

    ```{python}
    recipe = (
        Recipe(data=credit, target="Status")
        .add(StepDropCorr(s.numeric_features(), threshold=0.9))
        .add(StepNormalize(s.numeric_features()))
    )
    ```

    The untrained recipe shows the roles it carries and its steps with their
    selectors unresolved:

    ```{python}
    recipe
    ```

    `set_roles()` declares roles on the input columns, so it can be applied
    anywhere among the steps, including after them, with the same result (once
    a role operation is written before the first step, `set_roles()` keeps the
    order it was written in with it, see `set_roles()`):

    ```{python}
    late = (
        Recipe(data=credit)
        .add(StepDropCorr(s.numeric_features(), threshold=0.9))
        .add(StepNormalize(s.numeric_features()))
        .set_roles(target="Status")
    )
    late
    ```

    To change a role part way through the pipeline instead, use `add_role()`,
    `update_role()` or `remove_role()`, which take effect from where they are
    written.

    Parameters
    ----------
    steps
        Steps to start from, equivalent to chaining `add()` once per step.
        Default ``None`` (no steps).
    data
        A data frame whose column names are used to check role assignments as soon
        as they are made, and which backs `summary()` before fitting. Nothing
        is estimated from it and it is never modified, so a small sample with
        the right columns is enough. Default ``None``.
    target
        Column name or list of names to give the ``"target"`` role. Default
        ``None``.
    features
        Column name or list of names to give the ``"feature"`` role. With the
        default, ``None``, every column that is not a target or a case weight
        becomes a feature when the recipe is fitted, including columns given a
        custom role with `set_roles()`. Pass names to choose the features
        yourself and keep such columns out.
    case_weights
        Column name or list of names to give the ``"case_weights"`` role.
        Default ``None``.
    """

    #: The steps, in the order they run.
    steps: list[Step]
    #: The optional prototype data frame backs column-existence checks during role
    #: assignment and the pre-fit ``summary()``. It is never mutated, so
    #: clones share the reference. The lineage DAG is derived from it on
    #: demand rather than stored; see ``summary()``.
    _prototype_frame: Frame | None
    #: The roles declared by the constructor and by ``set_roles()``. Once a
    #: role operation is written before the first step, later ``set_roles()``
    #: calls are kept in order with it (as operations) and are not added here;
    #: the public ``roles`` property returns a copy with those operations applied.
    _declared_roles: Roles
    #: Pending transform-time requirement overrides, applied at ``fit()``.
    #: Each entry is a (target, transform_required) pair where target is a
    #: role name or a selector resolved against the training data frame.
    _role_req_overrides: list[tuple[str | FeaturesSelector, bool]]
    #: Positional role operations (`add_role`, `update_role`, `remove_role`),
    #: each tagged with the number of steps that precede it.
    _role_ops: list[_RoleOp]

    def __init__(
        self,
        steps: list[Step] | None = None,
        *,
        data: DataFrame | None = None,
        target: str | list[str] | None = None,
        features: str | list[str] | None = None,
        case_weights: str | list[str] | None = None,
    ):
        self.steps = _resolve_step_ids([], tuple(steps)) if steps is not None else []
        self._prototype_frame = _frame.from_native(data) if data is not None else None
        self._declared_roles = Roles()
        self._role_req_overrides = []
        self._role_ops = []
        _apply_role_assignments(
            self._declared_roles,
            {"target": target, "feature": features, "case_weights": case_weights},
            check_col=self._check_column_exists,
        )

    @property
    def roles(self) -> Roles:
        """The roles the recipe starts from, before any step runs.

        A read-only copy: the recipe is immutable, so calling ``add()`` or
        ``remove()`` on the returned roles raises (use `set_roles()` or the role
        operations for that, or ``copy()`` for a mutable copy).
        Role operations written before the first step, including `set_roles()`
        calls made after one, are applied to it; default features are not
        shown, so it holds only what was declared.
        """
        try:
            return self._leading_roles().copy(read_only=True)
        except ValueError:
            # A bad role operation surfaces when fitting; reading must not fail.
            return self._declared_roles.copy(read_only=True)

    def __repr__(self) -> str:
        """A human-readable, untrained view of the pipeline.

        Shows the declared role assignments and the steps with their unresolved
        selectors, each tagged ``[untrained]``. Column counts come from the
        prototype data frame when one was supplied; otherwise only the explicitly
        assigned roles are summarised. ``FittedRecipe`` has its own ``repr`` that
        shows resolved columns and the ``trained`` state instead.
        """
        n_columns = (
            len(_frame.columns(self._prototype_frame))
            if self._prototype_frame is not None
            else None
        )
        try:
            roles = self._leading_roles()
        except ValueError:
            # A bad role operation surfaces when fitting; printing must not fail.
            roles = self._declared_roles
        return _format_recipe(
            trained=False,
            roles=roles,
            steps=self.steps,
            n_columns=n_columns,
            role_ops=self._role_ops,
        )

    def _clone(self) -> "Recipe":
        """A shallow copy that preserves steps, prototype, roles, and overrides."""
        new = Recipe(steps=self.steps)
        new._prototype_frame = self._prototype_frame
        new._declared_roles = self._declared_roles.copy()
        new._role_req_overrides = list(self._role_req_overrides)
        new._role_ops = list(self._role_ops)
        return new

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Step hyperparameters keyed ``<step id>__<parameter>``.

        Follows scikit-learn's nested-parameter convention, so a
        `RecipeTransformer` exposes them as ``recipe__<step id>__<parameter>``
        to `GridSearchCV` and friends. Steps are addressed by ``id``, never by
        position. A step's ``id`` and ``columns`` are not tunable and are left
        out, as are steps with no tunable parameters.

        Parameters
        ----------
        deep
            Accepted for sklearn compatibility: every parameter lives inside
            a step, so ``deep=False`` returns an empty ``dict``. Default
            ``True``.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepDropCorr, StepNormalize
        from mlfeatures import Selectors as s
        ```

        A step's ``id`` is its default name (``"corr"`` for `StepDropCorr`),
        which the keys are built from:

        ```{python}
        recipe = (
            Recipe()
            .add(StepDropCorr(s.numeric_features(), threshold=0.9))
            .add(StepNormalize(s.numeric_features()))
        )
        recipe.get_params()
        ```
        """
        if not deep:
            return {}
        return {
            f"{step.id}__{name}": value
            for step in self.steps
            for name, value in step.get_params().items()
        }

    def set_params(self, **params: object) -> "Recipe":
        """Return a new recipe with step hyperparameters replaced.

        Keys are ``<step id>__<parameter>``, as returned by `get_params`. The
        affected steps are rebuilt (so their argument validation runs again) and
        the recipe's roles and prototype data carry over; ``self`` is left
        untouched.

        Use it to change a hyperparameter of a step already in the recipe, such
        as a threshold you are tuning, without respecifying the pipeline. To
        change which steps run, use `add` instead.

        Parameters
        ----------
        **params
            New values keyed ``<step id>__<parameter>``.

        Raises
        ------
        ValueError
            If a key is not of the form ``<step id>__<parameter>``, or names a
            step id the recipe does not have.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepDropCorr, StepNormalize
        from mlfeatures import Selectors as s
        ```

        ```{python}
        recipe = (
            Recipe()
            .add(StepDropCorr(s.numeric_features(), threshold=0.9))
            .add(StepNormalize(s.numeric_features()))
        )
        ```

        The original keeps its value; the returned recipe has the new one:

        ```{python}
        recipe.set_params(corr__threshold=0.5).get_params()
        ```
        """
        by_step: dict[str, dict[str, object]] = {}
        for key, value in params.items():
            step_id, sep, name = key.partition("__")
            if not sep or not name:
                raise ValueError(
                    f"invalid parameter {key!r} for Recipe; "
                    "use '<step id>__<parameter>'"
                )
            by_step.setdefault(step_id, {})[name] = value
        ids = [step.id for step in self.steps]
        unknown = [step_id for step_id in by_step if step_id not in ids]
        if unknown:
            raise ValueError(
                f"no step with id(s) {unknown!r} in this recipe; valid ids are {ids!r}"
            )
        new = self._clone()
        new.steps = [
            step.set_params(**by_step[step.id]) if step.id in by_step else step
            for step in self.steps
        ]
        return new

    def __sklearn_clone__(self) -> "Recipe":
        """Let `sklearn.base.clone` copy a recipe without rebuilding it from params.

        sklearn would otherwise call ``Recipe(**get_params(deep=False))``, which
        is not how a recipe is constructed. `Recipe` is immutable, so a
        shallow copy is a faithful clone.
        """
        return self._clone()

    def add(self, *steps: Step) -> "Recipe":
        """Return a new recipe with ``steps`` added after the existing ones.

        The recipe itself is never changed, so a recipe can be extended in
        several directions from a common starting point. Steps run in the
        order they were added.

        Step ids are kept unique within the recipe. A step that did not get an
        explicit ``id`` and whose default collides with an earlier step's is
        added under a suffixed id instead (``"dummy"``, then ``"dummy_2"``),
        so the two are addressable separately. Passing an ``id`` that a step
        already in the recipe uses raises ``ValueError``.

        Parameters
        ----------
        *steps
            The steps to add, in order.

        Examples
        --------
        Each ``add()`` returns a new recipe, so steps chain into a pipeline
        without mutating the original:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        ```

        ```{python}
        recipe = (
            Recipe()
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        recipe
        ```

        Adding the same kind of step twice keeps the ids distinct:

        ```{python}
        twice = recipe.add(StepNormalize(s.integer_features()))
        [step.id for step in twice.steps]
        ```
        """
        new = self._clone()
        new.steps = [*self.steps, *_resolve_step_ids(self.steps, steps)]
        return new

    def set_roles(
        self,
        *,
        target: str | list[str] | None = None,
        feature: str | list[str] | None = None,
        case_weights: str | list[str] | None = None,
        **custom: str | list[str],
    ) -> "Recipe":
        """Return a new recipe with the given role assignments added.

        Standard roles have dedicated keyword arguments; arbitrary custom roles
        may be passed as additional keywords. Each value is a column name or a
        list of names. Assignments are *added* to any roles the recipe already
        carries, so a column may end up with several roles at once; incompatible
        combinations (e.g. target + feature on one column) are rejected. When
        the recipe was built with a prototype ``data`` frame, every name is
        checked against it; without a prototype there is nothing to check
        against, so any name is accepted and typos surface later, when the
        recipe is fitted.

        Roles are what let a step say "all numeric features" without also
        picking up the target or case weights: selectors such as
        ``Selectors.numeric_features()`` only match columns carrying the
        ``feature`` role. Declare ``feature`` explicitly to keep other columns,
        such as an ID, out of those selections. A custom role marks columns for
        your own selectors, or for `update_role_requirements`.

        Roles apply to the columns of the input data, whatever the call's
        position among the steps. Columns that steps create get their roles
        from the step that creates them. ``set_roles()`` only adds roles; to
        add, change, or remove one part way through the pipeline, use
        `add_role`, `update_role` or `remove_role`. Next to those written
        before the first step, which also act on the input columns, it keeps
        the order it was written in, so
        ``update_role("x", new_role="id").set_roles(group="x")`` leaves ``x``
        with both roles, while the same two calls the other way round leave it
        with ``"id"`` only. A custom role can be selected in a step with
        ``Selectors.has_role()``.

        The ``roles`` property returns a copy of the roles the recipe starts
        from: what the constructor and ``set_roles()`` declared, with any role
        operation written before the first step applied (default features are
        not shown). Changing the copy does not change the recipe; `summary()`
        shows the same roles per column.

        Parameters
        ----------
        target
            Column(s) to assign the ``"target"`` role. Default ``None``.
        feature
            Column(s) to assign the ``"feature"`` role. Default ``None``.
        case_weights
            Column(s) to assign the ``"case_weights"`` role. Default ``None``.
        **custom
            Any further keywords assign arbitrary custom roles, the keyword
            being the role name (e.g. ``group="city"``). The constructor's
            plural ``features=`` is rejected here rather than silently
            becoming a custom role.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe
        from mlfeatures.data import credit
        ```

        Assigning roles returns a new recipe, leaving the original untouched:

        ```{python}
        recipe = Recipe(data=credit).set_roles(
            target="Status", feature=["Age", "Income", "Home"]
        )
        recipe
        ```

        Custom roles are just extra keywords, and stack with the roles a column
        already has:

        ```{python}
        recipe.set_roles(group="Home").roles.of("Home")
        ```
        """
        # ``**custom`` swallows any keyword, so the constructor's plural
        # spelling would otherwise create a bogus 'features' role in silence.
        if "features" in custom:
            raise ValueError(
                "set_roles() takes 'feature', not 'features'; "
                "the plural spelling is the Recipe() constructor's"
            )

        assignments: dict[str, str | list[str] | None] = {
            "target": target,
            "feature": feature,
            "case_weights": case_weights,
            **custom,
        }
        for role in assignments:
            if not role:
                raise ValueError(f"'role' must be a non-empty string; got {role!r}")
        if any(op.position == 0 for op in self._role_ops):
            # A role operation written before the first step already ran on the
            # declared roles, so these assignments have to come after it, as
            # written, rather than be folded into the declarations. They are
            # still checked now, by the same code as in the other branch, against
            # the roles the recipe starts from once those operations have run
            # (a bad earlier operation is reported when fitting, not here).
            # Without prototype data an update or removal can't be replayed, so
            # the starting roles would be wrong; fit() does the check then.
            replayable = self._prototype_frame is not None or all(
                op.position != 0 or (op.kind == "add" and op.names is not None)
                for op in self._role_ops
            )
            if replayable:
                try:
                    starting = self._leading_roles()
                except ValueError:
                    starting = self._declared_roles.copy()
                _apply_role_assignments(
                    starting, assignments, check_col=self._check_column_exists
                )
            new = self
            for role, columns in assignments.items():
                if columns is None:
                    continue
                names = [columns] if isinstance(columns, str) else list(columns)
                if names:
                    new = new._with_role_op("add", names, role, declared=True)
            return new._clone() if new is self else new

        new = self._clone()
        _apply_role_assignments(
            new._declared_roles, assignments, check_col=new._check_column_exists
        )
        return new

    def _with_role_op(
        self,
        kind: Literal["add", "update", "remove"],
        columns: Selector,
        role: str,
        old_role: str | None = None,
        *,
        declared: bool = False,
    ) -> "Recipe":
        """Record a role operation after the steps added so far.

        ``declared`` records a ``set_roles()`` assignment instead: it acts on
        the input columns, so it sits before the first step.
        """
        role_label = "new_role" if kind == "update" else "role"
        for label, value in ((role_label, role), ("old_role", old_role)):
            if value is None and label == "old_role":
                continue
            if not isinstance(value, str) or not value:  # pyright: ignore[reportUnnecessaryIsInstance]
                raise ValueError(f"'{label}' must be a non-empty string; got {value!r}")
        raw: object = columns
        if not isinstance(
            columns, (str, list, tuple, FeaturesSelector, NwSelector, PlExpr)
        ):
            raise TypeError(
                "'columns' must be a column name, a list of names, or a selector; "
                f"got {columns!r}"
            )
        if isinstance(columns, (list, tuple)) and not all(
            isinstance(v, str) for v in cast("list[object]", columns)
        ):
            raise TypeError(
                "'columns' must be a column name, a list of names, or a selector; "
                f"got {columns!r} (a list cannot mix names and selectors)"
            )
        plain: tuple[str, ...] | None = None
        if isinstance(columns, str):
            plain = (columns,)
        elif isinstance(columns, (list, tuple)):
            # Copied: the recipe is immutable, so a list the caller keeps
            # editing must not change which columns the operation names.
            copied = [
                str(v) for v in cast("list[object] | tuple[object, ...]", columns)
            ]
            raw = copied
            plain = tuple(copied)
        op = _RoleOp(
            position=0 if declared else len(self.steps),
            kind=kind,
            columns=Selectors.wrap(raw),
            role=role,
            old_role=old_role,
            names=plain,
            declared=declared,
        )
        # Before the first step the columns are the input columns, so plain
        # names can be checked against the prototype data right away.
        names = op.plain_names()
        if op.position == 0 and names is not None:
            for name in names:
                self._check_column_exists(name)
        new = self._clone()
        new._role_ops.append(op)
        return new

    def _leading_roles(self) -> Roles:
        """The declared roles with the role operations written before any step.

        Needs the prototype data to resolve selectors; without it only plain
        column names are applied.
        """
        roles = self._declared_roles.copy()
        frame = self._prototype_frame
        if frame is not None:
            lineage = Lineage.from_frame(frame, roles)
            _apply_leading_ops(
                self._role_ops, frame, roles, lineage, show_defaults=False
            )
            return roles
        for op in self._role_ops:
            if op.position != 0:
                continue
            names = op.plain_names()
            if names is None or op.kind != "add":
                continue
            for name in names:
                roles.add(name, op.role)
        return roles

    def _declared_targets(self) -> list[str]:
        """Target columns known before fitting: declared roles plus plain-name ops."""
        if self._prototype_frame is not None:
            # Resolve the leading operations (selectors included) against the data.
            try:
                return self._leading_roles().columns_with_role("target")
            except ValueError:
                # A bad operation is reported by fit() with its own context.
                return self._declared_roles.columns_with_role("target")
        targets = list(self._declared_roles.columns_with_role("target"))
        # Only operations before the first step act on the input columns, so
        # only those can change which input columns are targets; replay them
        # in order.
        for op in self._role_ops:
            names = op.plain_names()
            if op.position != 0 or not names:
                continue
            if op.kind == "add":
                if op.role == "target":
                    targets.extend(n for n in names if n not in targets)
                continue
            dropped = op.role if op.kind == "remove" else op.old_role
            if dropped is None or dropped == "target":
                targets = [t for t in targets if t not in names]
            if op.kind == "update" and op.role == "target":
                targets.extend(n for n in names if n not in targets)
        return targets

    def add_role(self, columns: Selector, *, role: str) -> "Recipe":
        """Return a new recipe with a role added at this point in the chain.

        Unlike `set_roles()`, which declares roles on the input columns whatever
        its position, a role operation takes effect where it is written: steps
        added after it see the new role, steps added before it do not. It can
        also reach columns that earlier steps created.

        The columns are resolved when the recipe is fitted, against the columns
        as the steps before it left them. A selection matching nothing changes
        nothing. Adding a role a column already has is harmless; an
        incompatible one (``"target"`` on a feature) raises when the recipe is
        fitted.

        Wherever the call is written, a column that is only a feature by
        default (see `Recipe`) and is made the ``"target"`` or
        ``"case_weights"`` is taken out of the features instead of raising or
        keeping both roles, like `set_roles()`. A feature role you declared
        yourself is never taken away, so it still conflicts with
        ``"target"``. Adding the ``"feature"`` role never changes which other
        columns are features.

        Parameters
        ----------
        columns
            Columns to give the role to. Accepts a single column name, a list of names, a
            mlfeatures selector such as ``Selectors.numeric_features()`` (see
            `Selectors` for the full set), or a narwhals or polars selector
            (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
            Selectors are resolved against the data frame as the steps before
            this one left it, when the recipe is fitted.
        role
            The role to add.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        Give ``Seniority`` a ``"group"`` role after the normalization, so the
        step before it is unaffected:

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add_role(s.starts_with("Senior"), role="group")
        )
        recipe
        ```

        Fitting applies the role, which the summary then shows:

        ```{python}
        fitted = recipe.fit(credit)
        ```

        ```{python}
        fitted.summary()
        ```
        """
        return self._with_role_op("add", columns, role)

    def update_role(
        self, columns: Selector, *, new_role: str, old_role: str | None = None
    ) -> "Recipe":
        """Return a new recipe with a role changed at this point in the chain.

        Takes effect where it is written (see `add_role()`). With ``old_role``
        only that role is swapped for ``new_role`` and the column's other roles
        stay; without it, all of a column's roles are replaced by ``new_role``.

        Parameters
        ----------
        columns
            Columns to change the role of. Accepts a single column name, a list of names, a
            mlfeatures selector such as ``Selectors.numeric_features()`` (see
            `Selectors` for the full set), or a narwhals or polars selector
            (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
            Selectors are resolved against the data frame as the steps before
            this one left it, when the recipe is fitted.
        new_role
            The role the columns end up with.
        old_role
            The role to replace. Every selected column must carry it, or
            fitting raises ``ValueError``. Default ``None`` (replace all roles).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        Normalize every numeric feature, then swap the ``"feature"`` role of
        ``Seniority`` for ``"id"``, so steps added after this call no longer
        select it:

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .update_role(s.starts_with("Senior"), new_role="id", old_role="feature")
        )
        recipe
        ```

        Fitting applies the change, which the summary then shows:

        ```{python}
        fitted = recipe.fit(credit)
        ```

        ```{python}
        fitted.summary()
        ```
        """
        return self._with_role_op("update", columns, new_role, old_role)

    def remove_role(self, columns: Selector, *, role: str) -> "Recipe":
        """Return a new recipe with a role removed at this point in the chain.

        Takes effect where it is written (see `add_role()`). Every selected
        column must carry the role, or fitting raises ``ValueError``. A column
        left with no role is kept, but role selectors no longer match it.

        Parameters
        ----------
        columns
            Columns to remove the role from. Accepts a single column name, a list of names, a
            mlfeatures selector such as ``Selectors.numeric_features()`` (see
            `Selectors` for the full set), or a narwhals or polars selector
            (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
            Selectors are resolved against the data frame as the steps before
            this one left it, when the recipe is fitted.
        role
            The role to remove.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .remove_role(s.starts_with("Senior"), role="feature")
        )
        recipe
        ```

        Fitting applies the change, which the summary then shows:

        ```{python}
        fitted = recipe.fit(credit)
        ```

        ```{python}
        fitted.summary()
        ```
        """
        return self._with_role_op("remove", columns, role)

    def update_role_requirements(
        self,
        role: str | FeaturesSelector,
        *,
        transform_required: bool,
    ) -> "Recipe":
        """Return a new recipe overriding whether columns are required at transform.

        By default every column the recipe was fitted on must be present in the
        data given to `FittedRecipe.transform()`, except the target and case
        weights. That includes columns that no step uses, such as an ID column
        with a custom role, which is rarely what you want: transforming
        without it raises. Mark it as not required, or drop it from the data
        before fitting. The requirement of every column is recorded in the
        ``transform_required`` mapping of `FittedRecipe.schema`.

        The ``feature`` and ``target`` roles are fixed: features are always
        required at transform and targets never are. A column that holds a
        custom role *and* the ``feature`` role stays required, so declare
        ``features`` explicitly when a column carries a custom role (see
        `Recipe`).

        Parameters
        ----------
        role
            A role name (e.g. ``"id"``), or a selector such as
            ``Selectors.starts_with("id_")`` resolved against the training
            data frame when the recipe is fitted. Passing ``"feature"`` or
            ``"target"`` raises ``ValueError``. A selector that happens to
            match a feature or target column leaves that column unchanged.
        transform_required
            ``True`` to require the columns at transform, ``False`` to allow
            them to be absent. When several overrides match a column, the last
            one wins.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        Give ``Seniority`` an ``"id"`` role, relax its requirement, and fit:

        ```{python}
        recipe = (
            Recipe(
                data=credit,
                target="Status",
                features=["Age", "Income", "Home"],
            )
            .set_roles(id="Seniority")
            .update_role_requirements("id", transform_required=False)
            .add(StepNormalize(s.numeric_features()))
        )
        fitted = recipe.fit(credit)
        ```

        New data without the ID column now transforms:

        ```{python}
        fitted.transform(credit.drop("Seniority").head(5))
        ```
        """
        if isinstance(role, str) and role in _NON_CONFIGURABLE_ROLES:
            raise ValueError(f"requirement for role {role!r} is not configurable")
        new = self._clone()
        new._role_req_overrides = [
            *self._role_req_overrides,
            (role, transform_required),
        ]
        return new

    def _resolve_role_req_overrides(
        self, frame: Frame, roles: Roles
    ) -> dict[str, bool]:
        """Resolve pending overrides to a column→requirement map (last wins)."""
        resolved: dict[str, bool] = {}
        for role, required in self._role_req_overrides:
            if isinstance(role, str):
                columns = roles.columns_with_role(role)
            else:
                columns = role.resolve(frame, roles)
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

        Each step is fitted on the output of the step before it, so a later
        step sees the columns as the earlier ones left them, and its selector
        is resolved against those columns. Steps that only act on training
        samples, such as row filters, are applied here, so later steps are
        fitted on the filtered rows, but they are skipped by
        `FittedRecipe.transform()`.

        Every call estimates everything from scratch: the recipe is not
        changed, it can be fitted again on other data, and each `FittedRecipe`
        is independent of the recipe and of every other fit. Returning a
        separate trained object keeps the untrained specification reusable
        (for example inside cross-validation, where it is fitted once per
        fold) and makes the object that holds the statistics the one you save
        and apply to new data. Missing values are handled by the individual
        steps, not by the recipe. The training data is not kept, so to get the
        transformed training data call `FittedRecipe.transform()` on it; there
        is no ``fit_transform()``. Note that ``transform()`` skips row filters,
        so it can return more rows than the steps were fitted on. To see which
        columns each step added or removed, use `FittedRecipe.summary()`.

        When no ``features`` were declared, every column that is not a target
        or a case weight is given the ``"feature"`` role (see `Recipe`).

        Parameters
        ----------
        data
            The training data frame.

        Raises
        ------
        StepError
            If a step fails while fitting or while transforming the training
            data during the fit; the error names the step and the columns it
            was working on.
        ValueError
            If a step would produce a column name that already exists, or a
            role operation (`add_role()`, `update_role()`, `remove_role()`, or
            `set_roles()` written after one) fails: its columns cannot be
            selected, a column lacks the role it is asked to give up, the
            result would hold incompatible roles, or it is positioned after a
            step the recipe does not have.
        ImportError
            If a step needs an optional dependency that is not installed.

        Examples
        --------
        ``fit()`` estimates each step's statistics on the training data and
        returns a `FittedRecipe` with the selectors resolved to concrete
        columns:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        fitted = recipe.fit(credit)
        fitted
        ```
        """
        # Optional dependencies are only needed to run a step, so they are
        # checked here (and in ``FittedRecipe.transform()``) rather than when a
        # step is added: building, inspecting, and serializing a recipe stay
        # possible in an environment that lacks them.
        for step in self.steps:
            step.check_required_pkgs()
        current = _frame.from_native(data)
        # Fit *copies* of the steps, never the recipe's own: fitting writes
        # estimates, ``trained``, and ``_empty_selection`` onto a step, so
        # sharing them would let a later fit overwrite an earlier
        # FittedRecipe's state. Deep, so a step whose ``fit()`` mutates its
        # state in place is isolated too: the guarantee must not depend on how
        # a step author wrote ``fit()``.
        steps = [copy.deepcopy(step) for step in self.steps]
        live_roles = self._declared_roles.copy()
        # The column-lineage DAG, seeded from the training data frame and updated
        # after each column-changing step. RowOp and
        # empty-selection steps make no schema change and are skipped below.
        # It exists before the first step so that role operations written there
        # can use provenance selectors; its roles are refreshed once those and
        # feature defaulting have run.
        lineage = Lineage.from_frame(current, live_roles)
        fitted_steps: list[Step] = []
        # Input columns a role operation written after the first step resolved
        # to: only their transform-time requirement can differ from the one
        # derived from the input roles below.
        touched: set[str] = set()
        # Input columns a step reads at transform, as that step saw their roles.
        needed: set[str] = set()
        track_requirements = any(op.position > 0 for op in self._role_ops)

        def run_role_ops(position: int) -> None:
            ops = [op for op in self._role_ops if op.position == position]
            for op in ops:
                touched.update(
                    op.apply(
                        current,
                        live_roles,
                        lineage,
                        {s.id: s for s in fitted_steps},
                        implicit,
                    )
                )
                lineage.refresh_roles(live_roles)

        # Role operations written before the first step act on the input
        # columns, so they run before the schema snapshot and behave like
        # ``set_roles()`` there; see ``_apply_leading_ops`` for how they
        # interact with feature defaulting.
        implicit = _apply_leading_ops(
            self._role_ops, current, live_roles, lineage, show_defaults=True
        )
        # Snapshot the training data into a schema and validate the incoming
        # data frame against it (all columns required at fit). The snapshot rides on
        # the FittedRecipe so transform() can re-check new data at the boundary.
        input_frame = current
        overrides = self._resolve_role_req_overrides(current, live_roles)
        schema = Schema.from_frame(current, live_roles, overrides)
        schema.check(current, context="fit")
        input_names = set(schema.transform_required)
        lineage.refresh_roles(live_roles)
        dangling = [op for op in self._role_ops if op.position > len(steps)]
        if dangling:
            raise ValueError(
                f"{dangling[0].describe()} is positioned after step "
                f"{dangling[0].position} but the recipe has {len(steps)} step(s)"
            )
        for position, step in enumerate(steps):
            if position > 0:
                run_role_ops(position)
            info = SchemaInfo.from_frame(current)
            info.roles = live_roles
            # Hand the live lineage DAG and the id→step map to selector
            # resolution so provenance selectors (has_source/has_ancestor) can
            # resolve; these are recipe-side state, not for steps' own logic.
            info.lineage = lineage
            info.producing_steps = {s.id: s for s in fitted_steps}
            # Empty selection ⇒ identity. Resolved here, once, at the recipe
            # boundary so the no-op is structural: a step is never fitted on a
            # zero-column selection and so cannot misread it (the failure mode
            # the per-step guards used to handle). Steps with no input selector
            # (``StepIntercept``, the ``ADDING`` prototype; ``StepFilter``, a
            # ``RowOp``; ``StepMutate``, the ``UNCONSTRAINED`` escape hatch) are
            # exempt: they have no ``columns`` to resolve and must always run.
            # They are detected structurally by the absence of a ``columns``
            # attribute rather than by enumerating their types.
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
            # Dtype validation, also at the recipe boundary and also before the
            # step runs: a step declares what it accepts, the framework enforces
            # it. Only at fit time: the schema
            # already pins dtypes at transform time.
            if has_selector:
                _check_accepted_dtypes(step, current, selected)
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
            if track_requirements and not isinstance(step, RowOp):
                # What the step reads is only known once it has been fitted (a
                # step without a selector reports it then). A row operation is
                # skipped at transform, so it reads nothing there.
                read = selected if has_selector else fitted.schema_in().columns
                needed.update(
                    column
                    for column in read
                    if column in input_names
                    and column_transform_required(column, live_roles, overrides)
                )
            if not isinstance(step, RowOp):
                # Centralized name-collision guard: reject produced columns that
                # clash with the live schema or with each other, before the step
                # runs (see Lineage.check_additions). Steps no longer self-check.
                lineage.check_additions(fitted.id, fitted.produced_columns())
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
                lineage.update(fitted, current, live_roles)
            fitted_steps.append(fitted)
        if steps:
            run_role_ops(len(steps))
        # A role operation written after a step can change the roles of an input
        # column, and with them whether transform() needs it: a column that
        # becomes case weights no longer does, one that becomes a feature now
        # does. So the requirement of every column such an operation touched is
        # derived again from its final roles, with the overrides resolved
        # against them (a column that gained a role takes that role's override).
        # A step that read the column while it was required keeps it required,
        # or a missing column would pass the schema check and then fail inside
        # the step.
        changed = touched & input_names
        if changed:
            final_overrides = self._resolve_role_req_overrides(input_frame, live_roles)
            required = dict(schema.transform_required)
            for column in changed:
                required[column] = column in needed or column_transform_required(
                    column, live_roles, final_overrides
                )
            schema = replace(schema, transform_required=required)
        return FittedRecipe(
            steps=fitted_steps, roles=live_roles, lineage=lineage, schema=schema
        )

    def summary(self) -> DataFrame:
        """The input columns, before any step runs, as a tidy data frame.

        One row per column, with its ``name``, ``dtype``, ``roles`` (a column
        with several roles lists them all in one cell), ``source`` (always
        ``"original"`` before fitting) and whether it is ``sparse``. Requires
        data: construct the recipe with ``Recipe(data=...)``, or fit it and
        use `FittedRecipe.summary()` for the columns the recipe produces.

        Use it to check that the roles you assigned landed on the columns you
        meant before fitting; it cannot show what the steps will add or remove.
        Role operations written before the first step are reflected; later ones
        take effect at fit time and appear in `FittedRecipe.summary()`.

        Raises
        ------
        ValueError
            If the recipe was constructed without ``data``.

        Examples
        --------
        The pre-fit summary describes the input columns and their roles before
        any step has run, so ``Home`` is still a single ``str`` column with
        source ``"original"``:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit
        ```

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        ```

        ```{python}
        recipe.summary()
        ```
        """
        if self._prototype_frame is None:
            raise ValueError(
                "summary() needs prototype data; construct the recipe with "
                "Recipe(data=...), or fit it and call FittedRecipe.summary()"
            )
        return Lineage.from_frame(
            self._prototype_frame, self._leading_roles()
        ).summary()


class FittedRecipe:
    """A trained pipeline: applies statistics learned at fit time to new data.

    A `FittedRecipe` is what `Recipe.fit()` returns. It contains all the information
    it needs to be able to apply the transformations to new data:

    ```{python}
    #| echo: false
    #| output: false
    from mlfeatures import Recipe, StepNormalize, StepDummy
    from mlfeatures import Selectors as s
    from mlfeatures.data import credit

    credit = credit.drop_nulls()
    ```

    ```{python}
    recipe = (
        Recipe(data=credit, target="Status")
        .add(StepNormalize(s.numeric_features()))
        .add(StepDummy(s.nominal_features()))
    )
    fitted = recipe.fit(credit)
    ```

    The `repr` shows the columns each step resolved to:

    ```{python}
    fitted
    ```

    Use `transform()` to apply the recipe to new data and `summary()` to
    inspect the resulting columns. A trained recipe can be saved and restored
    with `to_json()` and `from_json()` (or the `dict` and YAML equivalents).

    Attributes
    ----------
    steps
        The trained steps, in the order they run.
    roles
        The role assignments after fitting, including any columns the steps
        added.
    lineage
        How each output column was derived from the input columns, as a
        `Lineage`, or ``None`` if the recipe was built without one.
    schema
        The `Schema` of the training data, which `transform()` checks new
        data against, or ``None`` if the recipe was built without one.
    """

    def __init__(
        self,
        steps: list[Step],
        roles: Roles | None = None,
        lineage: Lineage | None = None,
        schema: Schema | None = None,
    ):
        self.steps = steps
        self.roles = roles if roles is not None else Roles()
        #: The column lineage after all steps have run, or ``None`` if the
        #: recipe was constructed without one.
        self.lineage = lineage
        #: A snapshot of the training-data schema, used to validate new data
        #: in ``transform()``, or ``None`` if absent.
        self.schema = schema

    def __repr__(self) -> str:
        """A human-readable, trained view of the fitted pipeline.

        Shows the post-fit role assignments and each step with the concrete
        columns its selector resolved to at fit time, tagged ``[trained]``. This
        is the resolved counterpart of ``Recipe``'s unresolved ``repr``.
        """
        n_columns = len(self.roles.all_columns()) if self.roles.all_columns() else None
        return _format_recipe(
            trained=True, roles=self.roles, steps=self.steps, n_columns=n_columns
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to a diffable, JSON-safe ``dict``. See `mlfeatures.to_dict` for the layout.

        Use the ``dict`` when you are embedding the recipe in a larger payload
        or handing it to your own writer; for a file, prefer `to_json` or
        `to_yaml`.

        Returns
        -------
        dict
            Plain data holding the roles, schema, lineage, and every step's
            hyperparameters and learned statistics.

        Raises
        ------
        TypeError
            If a step holds something that cannot be serialized, such as a raw
            polars or narwhals selector, a ``dateutil`` rule, or a function.

        Warns
        -----
        UserWarning
            If a step comes from a package that `mlfeatures.from_dict` refuses
            by default.
        """
        from mlfeatures import serialization

        return serialization.to_dict(self)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a diffable JSON string. See `mlfeatures.to_json`.

        JSON needs no optional dependency and is readable by any tool, so it is
        the default choice for storing a recipe next to a model.

        Parameters
        ----------
        indent
            Spaces of indentation per level. Defaults to 2.

        Returns
        -------
        str
            The JSON text. Load it again with `FittedRecipe.from_json`.

        Raises
        ------
        TypeError
            As for `FittedRecipe.to_dict`.

        Warns
        -----
        UserWarning
            As for `FittedRecipe.to_dict`.
        """
        from mlfeatures import serialization

        return serialization.to_json(self, indent=indent)

    def to_yaml(self) -> str:
        """Serialize to a diffable YAML string. See `mlfeatures.to_yaml`.

        Choose YAML over JSON when people will read or review the file, since
        the block layout diffs more cleanly. It needs the optional ``pyyaml``
        package.

        Returns
        -------
        str
            The YAML text. Load it again with `FittedRecipe.from_yaml`.

        Raises
        ------
        ImportError
            If the optional ``pyyaml`` package is not installed.
        TypeError
            As for `FittedRecipe.to_dict`.

        Warns
        -----
        UserWarning
            As for `FittedRecipe.to_dict`.
        """
        from mlfeatures import serialization

        return serialization.to_yaml(self)

    @classmethod
    def from_dict(
        cls, data: dict[str, object], *, trusted: Iterable[type] = ()
    ) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `FittedRecipe.to_dict` output. See `mlfeatures.from_dict`.

        Parameters
        ----------
        data
            The ``dict`` produced by `FittedRecipe.to_dict`.
        trusted
            Classes whose top-level packages the payload may also name classes
            from, for a step that is not registered. Defaults to none.

        Raises
        ------
        ValueError
            If the payload is malformed, has a ``format_version`` this
            mlfeatures does not read, or names a class outside ``mlfeatures``,
            the registered step plugins, and the packages of the ``trusted``
            classes.
        """
        from mlfeatures import serialization

        return serialization.from_dict(data, trusted=trusted)

    @classmethod
    def from_json(cls, text: str, *, trusted: Iterable[type] = ()) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `FittedRecipe.to_json` output. See `mlfeatures.from_json`.

        Parameters
        ----------
        text
            The JSON text produced by `FittedRecipe.to_json`.
        trusted
            Classes whose top-level packages the payload may also name classes
            from, for a step that is not registered. Defaults to none.

        Raises
        ------
        ValueError
            As for `FittedRecipe.from_dict`; text that is not valid JSON raises
            ``json.JSONDecodeError``, a ``ValueError`` subclass.
        """
        from mlfeatures import serialization

        return serialization.from_json(text, trusted=trusted)

    @classmethod
    def from_yaml(cls, text: str, *, trusted: Iterable[type] = ()) -> "FittedRecipe":
        """Reconstruct a fitted recipe from `FittedRecipe.to_yaml` output. See `mlfeatures.from_yaml`.

        Parameters
        ----------
        text
            The YAML text produced by `FittedRecipe.to_yaml`.
        trusted
            Classes whose top-level packages the payload may also name classes
            from, for a step that is not registered. Defaults to none.

        Raises
        ------
        ValueError
            As for `FittedRecipe.from_dict`.
        ImportError
            If the optional ``pyyaml`` package is not installed.
        """
        from mlfeatures import serialization

        return serialization.from_yaml(text, trusted=trusted)

    def minimize(self) -> "FittedRecipe":
        """Return a recipe that keeps only what `transform()` needs. See `mlfeatures.minimize`.

        Each step drops what it can: anything kept only for inspection after
        fitting, or only because fitting needed it. The result transforms new
        data identically and ``self`` is left unchanged. A step with nothing
        to drop is shared with ``self`` rather than copied.

        Call it before saving or shipping a recipe, so the artifact carries
        only what scoring needs. `FittedRecipe.to_dict` and the JSON and YAML
        writers save the recipe as it is and do not minimize it. No shipped
        step currently holds anything to drop, so today this returns an
        equivalent recipe.
        """
        from mlfeatures import serialization

        return serialization.minimize(self)

    def fit(self, data: DataFrame) -> "FittedRecipe":
        """Re-fitting an already-fitted recipe is a no-op: returns ``self``.

        The stored statistics are never re-estimated, so ``rec.fit().fit()`` is
        cheap and yields the same object. ``data`` is ignored, which differs
        from a scikit-learn estimator (whose ``fit()`` re-estimates), so this
        warns to make the discarded data visible. Fit the untrained ``Recipe``
        again to estimate from new data.
        """
        warnings.warn(
            "the recipe is already fitted; fit() is a no-op and the data passed "
            "was ignored, so no statistics were re-estimated; fit the untrained "
            "recipe to estimate from this data",
            stacklevel=2,
        )
        return self

    def transform(self, data: DataFrame) -> DataFrame:
        """Apply each trained step, in order, to new data.

        Every step applies the statistics stored at fit time. Steps that only
        act on training samples (such as row filters) are skipped, as are steps
        whose selection matched no columns. The result is a data frame of the same
        type as ``data``.

        Parameters
        ----------
        data
            The data frame to transform. Its columns and dtypes must match the
            training data; a mismatch raises an error naming the difference.

        Raises
        ------
        StepError
            If a step fails while transforming; the error names the step and
            the columns it was working on.

        Examples
        --------
        ``transform()`` applies the trained recipe to new data, returning a
        data frame with the columns the recipe learned at fit time:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        fitted = recipe.fit(credit)
        ```

        ```{python}
        result = fitted.transform(credit.head(5))
        result
        ```
        """
        # A deserialized recipe may be transformed in an environment that lacks
        # a package the one it was fitted in had.
        for step in self.steps:
            step.check_required_pkgs()
        current = _frame.from_native(data)
        if self.schema is not None:
            self.schema.check(current, context="transform")
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
        """The columns the recipe produces, as a tidy data frame.

        One row per output column, with its dtype, role, and the step that
        created it (``"original"`` for columns passed through from the input).

        Use it to see which columns the trained recipe will hand to a model and
        which step each came from. For where a single column came from, or what
        it descends from, use the `Lineage` in the ``lineage`` attribute.

        Raises
        ------
        RuntimeError
            If the recipe was constructed without a lineage.

        Examples
        --------
        After fitting, the summary reflects the transformed columns: the
        nominal columns have been expanded into dummy indicators (source
        ``"dummy"``) while the normalized numeric columns keep their feature
        role:

        ```{python}
        #| echo: false
        #| output: false
        from mlfeatures import Recipe, StepNormalize, StepDummy
        from mlfeatures import Selectors as s
        from mlfeatures.data import credit

        credit = credit.drop_nulls()
        ```

        ```{python}
        recipe = (
            Recipe(data=credit, target="Status")
            .add(StepNormalize(s.numeric_features()))
            .add(StepDummy(s.nominal_features()))
        )
        fitted = recipe.fit(credit)
        ```

        ```{python}
        fitted.summary()
        ```
        """
        if self.lineage is None:
            raise RuntimeError("this FittedRecipe was constructed without a lineage")
        return self.lineage.summary()
