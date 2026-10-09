"""The ``Step`` / ``RowOp`` contract.

These abstract base classes define the contract every feature-engineering step
codes against. They are transcribed from ``design/step-contract.md`` and are, for
now, importable stubs: the abstract methods raise `NotImplementedError`.
Selector resolution, content-derived ``id`` generation, ``StepError`` wrapping,
and the conformance test suite are implemented in later issues (#17, #18, #21).
"""

from __future__ import annotations

import inspect
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from mlfeatures.types import (
    AcceptedDtypes,
    ColSchema,
    Frame,
    KeepOriginals,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from typing_extensions import Self

#: Constructor arguments that identify a step rather than tune it: ``id`` is
#: the key tuning APIs address a step by, and ``columns`` is a selector.
_UNTUNABLE_PARAMS = frozenset({"id", "columns"})

#: Constructor arguments the framework stores under a different attribute name.
_PARAM_ATTRS = {"role": "output_role"}


def _step_type_label(step: "Step | type[Step]") -> str:
    """The R-style snake-case name for a step's type (e.g. ``step_normalize``).

    Derived from the class name: the ``Step`` prefix is dropped and the
    remaining CamelCase is converted to ``snake_case`` (``StepDropCorr`` →
    ``step_drop_corr``). Used by the printing methods in ``design/inspection.md``
    and by the polars accessor's parity test. Accepts either a step instance or a
    step class.
    """
    name = (step if isinstance(step, type) else type(step)).__name__
    if name.startswith("Step"):
        name = name[len("Step") :]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return f"step_{snake}"


class StepError(Exception):
    """A failure raised while fitting or transforming a step.

    The recipe wraps any exception raised from a step's ``fit()`` or
    ``transform()`` in a ``StepError`` and chains the original with ``raise ...
    from``, so the underlying traceback is preserved. The wrapper attaches the
    offending step's ``id``, its type, and the column(s) it was operating on, so
    a failure deep inside a pipeline says *which* step broke without the user
    having to read the whole stack.

    Attributes
    ----------
    step_id:
        The ``id`` of the step that failed.
    step_type:
        The step's snake-case type name (e.g. ``'step_normalize'``).
    columns:
        The column(s) the step was operating on, or ``[]`` if unknown.
    phase:
        ``'fit'`` or ``'transform'`` — the lifecycle stage the failure occurred in.
    """

    def __init__(
        self,
        step_id: str,
        step_type: str,
        columns: list[str],
        phase: str,
    ):
        self.step_id = step_id
        self.step_type = step_type
        self.columns = columns
        self.phase = phase
        message = f"step {step_id!r} ({step_type}) failed during {phase}"
        if columns:
            message += f"; column(s) {columns!r}"
        super().__init__(message)

    @classmethod
    def wrap(cls, step: "Step", columns: list[str], phase: str) -> "StepError":
        """Build a ``StepError`` for ``step`` from its id, type, and ``columns``."""
        return cls(step.id, _step_type_label(step), columns, phase)


class Step(ABC):
    """Abstract base class for a feature-engineering step.

    A step is constructed declaratively, fitted against training data to estimate
    and store statistics, and then applied to new data.

    **Everything a step needs is computed at fit time.** Whatever a step has to
    learn from data — statistics, level sets, a basis, the list of columns to
    drop — is estimated once from the training frame and stored on the step.
    ``transform()`` only applies that stored state; it never re-estimates
    anything, however different the new data looks. This is what makes a fitted
    recipe reproducible and free of train-to-serve skew, and it holds for every
    step, so individual steps do not restate it.

    A step's ``columns`` selector is resolved once, against the training
    frame's schema, when the recipe is fitted. A selector that matches nothing
    is not an error: the step is kept in the fitted recipe for inspection but
    never fitted and never run, making it the identity transform.

    What a step promises about the frame it returns is *which* columns are
    present and what they are named, not where they sit. Column position is an
    implementation detail that may differ between steps and backends, so
    downstream code should select columns by name rather than by position.

    ``Step`` subclasses always run at transform time; `RowOp` subclasses never
    run on new data. There is intentionally no ``skip`` flag — whether a step
    runs at transform time is determined structurally by its type.

    A subclass' ``__init__`` stores its own arguments and then calls
    ``super().__init__()`` to hand the framework-owned ones to the base class:

    ```python
    class StepMaxAbs(Step):
        mode = Mode.ONE_TO_ONE

        def __init__(self, columns: Selector, *, id: str | None = None):
            self.columns = columns
            super().__init__(id, "max_abs")
    ```

    See ``__init__`` for the arguments it takes.
    """

    #: Stable identifier, defaulting to the step type's short name
    #: (user-overridable). All inspection and tuning APIs key on ``id``, never
    #: on position. Ids are unique within a recipe: a default id that would
    #: collide with an earlier step's is suffixed (``"dummy"``, ``"dummy_2"``),
    #: while an explicitly passed duplicate is rejected. See ``__init__``.
    id: str
    #: Whether ``id`` was passed by the caller rather than defaulted. Only the
    #: defaulted ones may be renamed to resolve a collision.
    _id_explicit: bool = False
    #: Whether this step has been fitted. Set by ``recipe.fit()`` after
    #: ``step.fit()`` returns — not by the step itself.
    trained: bool
    #: The shape of this step's input/output relationship. Declared by every
    #: concrete step; the framework reads it for lineage and to decide whether
    #: ``keep_original_cols`` is fixed or configurable.
    mode: "ClassVar[Mode]"
    #: Backing store for the read-only ``keep_original_cols`` property.
    _keep_original_cols: bool
    #: Role assigned to new columns produced by this step.
    output_role: str | None
    #: The step's primary input selector. Present on every selector-taking step
    #: (steps with no input selector, e.g. ``StepIntercept``, do not set it and
    #: are special-cased by the recipe so ``resolve_selection`` is never called
    #: on them).
    columns: Selector
    #: Set ``True`` by ``recipe.fit()`` when the step's primary selection
    #: resolves to zero columns. A marked step is the identity transform: it is
    #: kept in the fitted pipeline for inspection but never fitted and never run.
    #: This makes "empty selection ⇒ no-op" structural rather than a per-step
    #: convention — mirroring how `RowOp` makes the transform-time skip
    #: structural. Defaults to ``False``.
    _empty_selection: bool = False
    #: What ``fit()`` estimated from the training data, or ``None`` before
    #: fitting (and for steps that estimate nothing). Each step narrows this to
    #: its own payload type.
    _estimates: Any
    #: Schema of the columns the step consumed, captured by ``fit()``; ``None``
    #: before fitting and for steps that consume no columns.
    _schema_in: ColSchema | None
    #: Schema of the columns the step produced, captured by ``fit()``; ``None``
    #: before fitting and for steps that produce none.
    _schema_out: ColSchema | None
    #: Map from each column the step adds to its parent input columns, captured
    #: by ``fit()``; ``None`` before fitting and for steps that add no columns.
    _output_parents: dict[str, list[str]] | None

    def __init__(
        self,
        id: str | None,
        default_id: str,
        *,
        keep_original_cols: bool | None = None,
        output_role: str | None = None,
    ) -> None:
        """Initialize the attributes every step shares.

        A concrete step stores its own arguments, then calls this with the
        framework-owned ones; it never assigns ``id``, ``trained``,
        ``keep_original_cols``, or ``output_role`` directly. Routing ``id``
        through here is what lets the recipe tell a defaulted id from a
        caller-supplied one, and that distinction is what lets `Recipe` silently
        disambiguate two same-type steps while still rejecting an explicit
        duplicate.

        Parameters
        ----------
        id
            The ``id`` the caller passed to the step, or ``None`` if they
            omitted it.
        default_id
            The step type's short name, used as ``id`` when the caller passed
            none (e.g. ``"normalize"``).
        keep_original_cols
            Whether the step retains its source columns alongside any columns it
            produces. Only meaningful for a step whose `Mode` makes it
            configurable (``ONE_TO_MANY``, ``MANY_TO_MANY``), which forwards its
            own constructor argument here. Every other mode has a fixed value,
            derived from the mode; such a step omits the argument, and passing
            anything but that fixed value is an error. Defaults to ``None``,
            meaning "not stated".
        output_role
            Role assigned to the columns the step produces, or ``None`` for a
            step that produces no new columns. Defaults to ``None``.
        """
        self.id = default_id if id is None else id
        self._id_explicit = id is not None
        self.trained = False
        self._keep_original_cols = self._resolve_keep_original_cols(keep_original_cols)
        self.output_role = output_role
        self._estimates = None
        self._schema_in = None
        self._schema_out = None
        self._output_parents = None

    def _resolve_keep_original_cols(self, requested: bool | None) -> bool:
        """The ``keep_original_cols`` this step's `Mode` allows it to have."""
        mode = self._declared_mode()
        fixed = KeepOriginals.for_mode(mode).fixed_value()
        if fixed is None:
            # A configurable step always forwards its own argument, which has
            # its own ``False`` default, so ``None`` cannot reach here.
            return bool(requested)
        if requested is not None and requested != fixed:
            raise ValueError(
                f"{type(self).__name__} has mode {mode.value!r}; "
                f"'keep_original_cols' is fixed at {fixed!r} and cannot be "
                f"set to {requested!r}"
            )
        return fixed

    def _declared_mode(self) -> Mode:
        """This step class's `Mode`, with a pointed error when it is missing."""
        mode = getattr(type(self), "mode", None)
        if not isinstance(mode, Mode):
            raise TypeError(
                f"{type(self).__name__} must declare a 'mode' class attribute "
                "naming one of the 'Mode' members"
            )
        return mode

    @property
    def keep_original_cols(self) -> bool:
        """Whether to retain the source columns alongside any produced columns.

        Read-only: the value is settled when the step is constructed, and for
        every mode but ``ONE_TO_MANY`` and ``MANY_TO_MANY`` it is fixed by the
        mode rather than chosen. Reassigning it would silently break the step,
        so rebuild the step instead of mutating it.
        """
        return self._keep_original_cols

    @classmethod
    def _init_param_names(cls) -> list[str]:
        """Names of the concrete step's ``__init__`` parameters, in signature order."""
        return [
            name
            for name, param in inspect.signature(cls.__init__).parameters.items()
            if name != "self"
            and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]

    def _param_value(self, name: str) -> object:
        """The instance attribute holding the ``__init__`` argument ``name``."""
        return getattr(self, _PARAM_ATTRS.get(name, name))

    def get_params(self) -> dict[str, object]:
        """The step's tunable hyperparameters, keyed by constructor argument name.

        Read from the instance attribute of the same name as each ``__init__``
        parameter (``role`` is stored as ``output_role``). ``id`` and
        ``columns`` are not tunable and are left out.
        """
        return {
            name: self._param_value(name)
            for name in self._init_param_names()
            if name not in _UNTUNABLE_PARAMS
        }

    def set_params(self, **params: object) -> "Self":
        """Return a new untrained step with ``params`` replaced.

        The step is rebuilt through ``__init__``, so the usual argument
        validation runs again; ``self`` is left untouched. The new step keeps
        this step's ``id`` and ``columns``.
        """
        tunable = self.get_params()
        unknown = sorted(set(params) - set(tunable))
        if unknown:
            raise ValueError(
                f"invalid parameter(s) {unknown!r} for {type(self).__name__}; "
                f"valid parameter(s) are {sorted(tunable)!r}"
            )
        kwargs: dict[str, Any] = {
            name: self._param_value(name)
            for name in self._init_param_names()
            if name in _UNTUNABLE_PARAMS
        }
        kwargs.update(tunable)
        kwargs.update(params)
        new = type(self)(**kwargs)
        new._id_explicit = self._id_explicit
        return new

    @abstractmethod
    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve selectors, estimate and store statistics, return the fitted step.

        Must be idempotent given the same training data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, new_data: Frame) -> Frame:
        """Apply stored statistics to ``new_data``. Never re-estimates."""
        raise NotImplementedError

    @abstractmethod
    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        raise NotImplementedError

    @abstractmethod
    def schema_out(self) -> ColSchema:
        """Schema of the columns this step produced. Known after ``fit()``."""
        raise NotImplementedError

    def output_parents(self) -> dict[str, list[str]]:
        """Map each column this step *produces* to its parent input column(s).

        Known after ``fit()``. The keys are exactly the columns this step adds to
        the frame; each value lists the input columns that column descends from.
        The recipe consults this to populate the column-lineage DAG (see
        ``design/lineage.md``) and validates the keys against the post-step frame.

        The default returns ``{}`` — correct for steps that create no new columns
        (in-place ``ONE_TO_ONE`` such as ``step_normalize``; ``REMOVING``;
        ``ROW_OP``). Steps that add columns override this to declare per-output
        provenance: ``ONE_TO_MANY`` (each ``a1, a2 ← a``), ``MANY_TO_MANY`` (every
        output ← the whole input set, as in ``step_pca`` and
        ``step_class_distance``), and ``ADDING`` (``step_intercept``, whose column
        has no parents and so maps to ``[]``).
        """
        return {}

    def produced_columns(self) -> list[str]:
        """Ordered names of the columns this step *adds* to the frame.

        Known after ``fit()``. Unlike ``output_parents()`` (a name→parents map,
        so its keys are necessarily unique), this is an ordered list that
        *preserves duplicates*: a step that would emit two columns with the same
        name reports that name twice. The recipe validates this list for
        uniqueness — both against the existing schema and within the step
        itself — *before* the step runs, which is the single centralized
        name-collision check every column-adding step relies on. Validating up
        front means the error is clean on every backend (polars raises a
        backend-specific error when two columns share a name; pandas silently
        keeps only one), rather than a confusing failure or a silently dropped
        column mid-transform.

        The default derives the list from ``output_parents()`` keys, correct for
        every step whose outputs cannot collide by construction. Steps with a
        free-form naming scheme that *could* collide (``step_dummy`` /
        ``step_one_hot``, whose ``{col}_{level}`` names can clash across source
        columns or under a custom ``name_fn``) override this to preserve any
        duplicate names so the recipe can reject them.
        """
        return list(self.output_parents())

    @abstractmethod
    def accepts_dtypes(self) -> AcceptedDtypes:
        """Dtypes this step is valid to apply to.

        Return a set of narwhals dtype classes (`NUMERIC_DTYPES` is ready-made
        for numeric-only steps), or `ANY_DTYPE` to skip dtype validation
        entirely. The recipe checks the resolved selection's dtypes against this
        set before ``fit()`` runs and raises a ``TypeError`` naming the step, the
        offending column(s), and their dtypes, so a step never sees a column it
        declared itself invalid for. The check does not run again at
        ``transform()`` time: the schema snapshot already pins new data's dtypes
        to what was seen at fit.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def produces_sparse(self) -> bool:
        """Whether this step produces sparse output."""
        raise NotImplementedError

    def resolve_selection(self, data: Frame, info: SchemaInfo) -> list[str]:
        """Resolve this step's ``columns`` selector to concrete column names.

        This is the supported way for a step to find out which columns it was
        pointed at. Call it at the top of ``fit()`` and select from the frame
        with the names it returns; narwhals' own ``frame.schema`` then gives the
        dtypes:

        ```python
        def fit(self, training_data, info):
            names = self.resolve_selection(training_data, info)
            selected = training_data.select(names)
            dtypes = {name: str(dt) for name, dt in selected.schema.items()}
            ...
        ```

        The default resolves ``self.columns``, which is right for every
        selector-taking step, including ones that take a second selector for a
        different purpose: the primary input selector is always ``columns``.
        Override only if a step's primary input is named something else.

        Takes ``data`` as well as ``info`` because narwhals and polars selectors
        resolve against the frame's schema rather than against the schema
        snapshot alone.

        The recipe also calls this before ``fit()`` so that a selection matching
        no columns makes the step the identity transform; a step that resolves
        its own selection therefore never has to handle the empty case itself.
        """
        from mlfeatures import _frame

        return _frame.select(
            data, self.columns, info.roles, info.lineage, info.producing_steps
        ).columns

    def update_roles(self, roles: Roles) -> None:
        """Update ``roles`` to reflect the column changes made by this step.

        Called by the recipe after each step is fitted and transformed. The
        default is a no-op; steps that add, remove, or rename columns override
        this to keep roles in sync with the live schema.
        """

    def required_pkgs(self) -> list[str]:
        """Third-party packages required to run this step. Defaults to none.

        Each entry is a requirement string, either a bare name (``"numpy"``) or
        one carrying a version constraint (``"scikit-learn>=1.3"``). The recipe
        checks the declared packages with `check_required_pkgs` when it is
        fitted and when a fitted recipe transforms data, so a missing optional
        dependency is reported as an actionable install hint instead of an
        ``ImportError`` from somewhere inside the step. Only running a step needs
        the package; building, inspecting, and serializing a recipe do not.
        """
        return []

    def check_required_pkgs(self) -> None:
        """Raise if any package this step declares in `required_pkgs` is missing.

        The error names the step and the missing package(s) and suggests the
        ``pip install`` that fixes it. Version constraints in a requirement are
        reported but not enforced.
        """
        from mlfeatures._dependencies import require_packages

        requirements = self.required_pkgs()
        if requirements:
            require_packages(
                requirements,
                purpose=f"by step {self.id!r} ({_step_type_label(self)})",
            )

    def to_state(self) -> dict[str, object]:
        """Serialize this step's fields to a diffable, JSON-safe ``dict``.

        Every hyperparameter and learned statistic already lives as a plain
        instance attribute, so the default encodes ``self.__dict__`` through the
        shared value codec -- no per-step boilerplate. The step class itself is
        recorded separately (by qualified name) by `mlfeatures.serialization`. See
        ``design/serialization.md``.
        """
        from mlfeatures.serialization import (
            _encode,  # pyright: ignore[reportPrivateUsage]
        )

        return {k: _encode(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "Self":
        """Reconstruct a step from the ``dict`` produced by `to_state`.

        Bypasses ``__init__`` and restores the decoded fields directly. Any
        attribute missing from ``state`` (e.g. a field added in a newer version)
        keeps whatever a bare instance would carry, so within a major version an
        older payload loads cleanly without per-field shims.
        """
        from mlfeatures.serialization import (
            _decode,  # pyright: ignore[reportPrivateUsage]
        )

        step = cls.__new__(cls)
        for key, value in state.items():
            # ``keep_original_cols`` used to be a plain attribute and is now a
            # read-only property over ``_keep_original_cols``; redirect the old
            # key so a payload written before the change still loads.
            if key == "keep_original_cols":
                key = "_keep_original_cols"
            setattr(step, key, _decode(value))
        return step

    def minimize(self) -> "Self":
        """Return a copy of this step with fit-only artifacts stripped.

        The default is a no-op: it returns ``self`` unchanged. Steps that retain
        training artifacts only needed at fit time (e.g. a future KNN imputer's
        neighbor matrix) override this to return a lighter copy that still
        transforms identically. See ``recipe.minimize`` / ``design/serialization.md``.
        """
        return self

    def _describe_selection(self) -> str:
        """A short description of what columns this step acts on, for printing.

        Untrained: the unresolved selector (``all_numeric_features()``,
        ``['age']``). Trained: the concrete columns the selector resolved to at
        fit time, mirroring the resolved/unresolved distinction in
        ``design/inspection.md``. A column-adding step additionally shows the
        columns it produced (``city → city_LA, city_NYC``) so the expansion is
        visible once fitted. A trained identity step (empty selection)
        and a step that consumes no columns (e.g. ``step_intercept``) describe
        themselves explicitly rather than showing an empty list.
        """
        if self.trained:
            if self._empty_selection:
                return "<empty selection>"
            try:
                consumed = self.schema_in().columns
            except Exception:  # pragma: no cover - defensive: pre-fit / odd steps
                consumed = None
            if consumed is not None:
                produced = self.produced_columns()
                consumed_str = ", ".join(consumed)
                if produced and consumed_str:
                    return f"{consumed_str} → {', '.join(produced)}"
                if produced:
                    return ", ".join(produced)
                return consumed_str
        columns = getattr(self, "columns", None)
        return repr(columns) if columns is not None else ""

    def _repr_head(self) -> str:
        """The ``step_name(selection)`` half of the printed line (no state marker)."""
        return f"{_step_type_label(self)}({self._describe_selection()})"

    def __repr__(self) -> str:
        state = "trained" if self.trained else "untrained"
        return f"{self._repr_head()}  [{state}]"


class RowOp(Step):
    """Row-altering steps.

    ``transform()`` is a no-op on new data — these steps only take effect during
    ``fit()`` (e.g. filtering or shuffling training rows). The behaviour is
    structural: because the framework dispatches on type, a ``RowOp`` can never
    accidentally run at transform time.

    The mode follows from the base class, so a ``RowOp`` subclass does not
    declare one.
    """

    mode = Mode.ROW_OP
