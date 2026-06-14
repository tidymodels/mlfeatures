"""The ``Step`` / ``RowOp`` contract.

These abstract base classes define the contract every feature-engineering step
codes against. They are transcribed from ``design/step-contract.md`` and are, for
now, importable stubs: the abstract methods raise `NotImplementedError`.
Selector resolution, content-derived ``id`` generation, ``StepError`` wrapping,
and the conformance test suite are implemented in later issues (#17, #18, #21).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from shoez.types import AcceptedDtypes, ColSchema, Frame, Roles, SchemaInfo, Selector

if TYPE_CHECKING:
    from typing_extensions import Self


class Step(ABC):
    """Abstract base class for a feature-engineering step.

    A step is constructed declaratively, fitted against training data to estimate
    and store statistics, and then applied to new data. ``Step`` subclasses always
    run at transform time; `RowOp` subclasses never run on new data. There
    is intentionally no ``skip`` flag — whether a step runs at transform time is
    determined structurally by its type.
    """

    #: Stable, content-derived identifier (user-overridable). All inspection and
    #: tuning APIs key on ``id``, never on position.
    id: str
    #: Whether this step has been fitted. Set by ``recipe.fit()`` after
    #: ``step.fit()`` returns — not by the step itself.
    trained: bool
    #: Whether to retain the source columns alongside any produced columns.
    keep_original_cols: bool
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
        ``design/schema.md``) and validates the keys against the post-step frame.

        The default returns ``{}`` — correct for steps that create no new columns
        (in-place ``ONE_TO_ONE`` such as ``step_normalize``; ``REMOVING``;
        ``ROW_OP``). Steps that add columns override this to declare per-output
        provenance: ``ONE_TO_MANY`` (each ``a1, a2 ← a``), ``MANY_TO_MANY`` (every
        output ← the whole input set), and ``ADDING`` steps whose outputs derive
        from inputs (``step_class_distance``) or from nothing (``step_intercept``,
        whose column maps to ``[]``).
        """
        return {}

    @abstractmethod
    def accepts_dtypes(self) -> AcceptedDtypes:
        """Dtypes this step is valid to apply to.

        Checked against the schema before ``fit()`` runs. Return
        `ANY_DTYPE` to skip dtype validation entirely.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def produces_sparse(self) -> bool:
        """Whether this step produces sparse output."""
        raise NotImplementedError

    def resolve_selection(self, data: Frame, info: SchemaInfo) -> list[str]:
        """Resolve this step's primary input selector to concrete column names.

        Called by ``recipe.fit()`` *before* ``fit()`` so the recipe can detect
        an empty selection and treat the step as the identity transform (see
        ``_empty_selection``). The default resolves ``self.columns`` against the
        current frame, which is correct for every selector-taking step —
        including multi-selector steps like ``StepClassDistance`` whose primary
        (feature) selector is ``columns``. Steps with no input selector are
        special-cased by the recipe and never reach this method.

        Takes ``data`` (not just ``info``) because narwhals / polars selectors
        resolve against the frame's schema.
        """
        from shoez import _frame

        return _frame.select(
            data, self.columns, info.roles, info.schema, info.producing_steps
        ).columns

    def update_roles(self, roles: Roles) -> None:
        """Update ``roles`` to reflect the column changes made by this step.

        Called by the recipe after each step is fitted and transformed. The
        default is a no-op; steps that add, remove, or rename columns override
        this to keep roles in sync with the live schema.
        """

    def required_pkgs(self) -> list[str]:
        """Third-party packages required to run this step. Defaults to none."""
        return []


class RowOp(Step):
    """Row-altering steps.

    ``transform()`` is a no-op on new data — these steps only take effect during
    ``fit()`` (e.g. filtering or shuffling training rows). The behaviour is
    structural: because the framework dispatches on type, a ``RowOp`` can never
    accidentally run at transform time.
    """
