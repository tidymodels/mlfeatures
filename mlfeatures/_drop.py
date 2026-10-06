"""Shared fit/transform/schema logic for the REMOVING-mode "drop" steps.

`StepDrop`, `StepDropCorr`, `StepDropNearZeroVariance`, and
`StepDropZeroVariance` differ only in which columns they decide to drop:
``fit()`` resolves the selection and calls the step's own drop-candidate
rule, and ``transform()``/``schema_in()``/``schema_out()``/``update_roles()``
follow purely from the resulting column list, with nothing step-specific
left. This module holds that shared shape as a base class; each step's own
module supplies only its `ColumnDropStep._drop_candidates` rule, its own
``__init__`` (for its own hyperparameters), ``accepts_dtypes()``, and its
class docstring.

Not part of the public API: only the ``mlfeatures.step_drop*`` modules import
from here.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from mlfeatures import _frame
from mlfeatures.step import Step
from mlfeatures.types import ColSchema, Frame, Mode, Roles, SchemaInfo, Selector

if TYPE_CHECKING:
    from typing_extensions import Self


class ColumnDropStep(Step):
    """Shared REMOVING-mode column-drop lifecycle.

    A subclass implements `_drop_candidates` -- the rule deciding which
    resolved columns to drop -- plus its own ``__init__`` (for any
    hyperparameters), ``accepts_dtypes()``, and class docstring; every other
    lifecycle method here is shared.
    """

    mode = Mode.REMOVING

    def __init__(
        self, columns: Selector, *, id: str | None = None, default_id: str
    ) -> None:
        # Held verbatim. Interpreting a column spec (name, list, or selector) is
        # selection logic that belongs to the frame boundary, not the step:
        # ``fit()`` hands it to ``_frame.select``, which resolves it at fit time.
        self.columns = columns
        super().__init__(id, default_id)
        # Populated by ``fit()``: the resolved names to drop.
        self._estimates: list[str] | None = None
        self._schema_in: ColSchema | None = None

    @abstractmethod
    def _drop_candidates(self, selected: Frame) -> list[str]:
        """The columns to drop, computed from the resolved selection."""
        raise NotImplementedError

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the selection, decide what to drop, and store the decision."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = self._drop_candidates(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=list(selected.columns), dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Drop the columns chosen at fit time. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError(f"{type(self).__name__}.transform() called before fit()")
        return _frame.apply_drop(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError(f"{type(self).__name__}.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the columns this step produced — none; it only removes."""
        if self._estimates is None:
            raise RuntimeError(
                f"{type(self).__name__}.schema_out() called before fit()"
            )
        return ColSchema(columns=[])

    def update_roles(self, roles: Roles) -> None:
        if self._estimates:
            for col in self._estimates:
                for role in list(roles.of(col)):
                    roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """A column-drop step never produces sparse output."""
        return False
