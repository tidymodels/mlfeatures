"""The ``Step`` / ``RowOp`` contract.

These abstract base classes define the contract every feature-engineering step
codes against. They are transcribed from ``design/step-contract.md`` and are, for
now, importable stubs: the abstract methods raise :class:`NotImplementedError`.
Selector resolution, content-derived ``id`` generation, ``StepError`` wrapping,
and the conformance test suite are implemented in later issues (#17, #18, #21).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from shoez.types import AcceptedDtypes, ColSchema, DataFrame, SchemaInfo


class Step(ABC):
    """Abstract base class for a feature-engineering step.

    A step is constructed declaratively, fitted against training data to estimate
    and store statistics, and then applied to new data. ``Step`` subclasses always
    run at transform time; :class:`RowOp` subclasses never run on new data. There
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

    @abstractmethod
    def fit(self, training_data: DataFrame, info: SchemaInfo) -> "Step":
        """Resolve selectors, estimate and store statistics, return the fitted step.

        Must be idempotent given the same training data.
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, new_data: DataFrame) -> DataFrame:
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

    @abstractmethod
    def accepts_dtypes(self) -> AcceptedDtypes:
        """Dtypes this step is valid to apply to.

        Checked against the schema before ``fit()`` runs. Return
        :data:`~shoez.types.ANY_DTYPE` to skip dtype validation entirely.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def produces_sparse(self) -> bool:
        """Whether this step produces sparse output."""
        raise NotImplementedError

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
