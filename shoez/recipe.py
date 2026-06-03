"""The :class:`Recipe` lifecycle driver.

A minimal, vertical-slice implementation of the ``specify → fit → transform``
lifecycle described in ``design/lifecycle.md``. It exercises the step contract
end to end while deliberately omitting subsystems that are not yet built:
selectors, roles, the ptype snapshot, dtype pre-checks, and the lineage DAG.

Steps are pure: calling ``.step_*()`` returns a *new* ``Recipe`` with the step
appended — it never touches data. ``fit()`` runs each step's ``fit`` on the
training data in order, feeding the transformed output of one step into the next
(``RowOp`` steps are fitted but not applied in the fit loop). ``transform()``
applies the fitted steps to new data, skipping ``RowOp`` steps entirely.
"""

from __future__ import annotations

from shoez import _frame
from shoez._formula import parse_formula
from shoez.step import RowOp, Step
from shoez.types import DataFrame, Roles, SchemaInfo

__all__ = ["Recipe", "FittedRecipe"]


class Recipe:
    """A declarative, unfitted pipeline of feature-engineering steps.

    Steps are first-class :class:`~shoez.step.Step` objects (Option A in
    ``design/interfaces.md``). Assemble a recipe by passing them at construction::

        Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)])

    or with the immutable :meth:`add` builder, which returns a new recipe each
    call (preserving the purity guarantee in ``design/lifecycle.md``)::

        Recipe().add(StepDropCorr(["a", "b", "c"], threshold=0.9))

    A prototype ``data`` frame may be supplied to capture the column names used
    when assigning roles. Roles are declared either explicitly with
    :meth:`set_roles` or via the minimal ``formula`` sugar (``"y ~ ."``,
    ``"y ~ x1 + x2"``); the formula is just shorthand for an outcome/predictor
    split, never a richer mechanism.
    """

    def __init__(
        self,
        steps: list[Step] | None = None,
        *,
        data: DataFrame | None = None,
        formula: str | None = None,
    ):
        self.steps: list[Step] = list(steps) if steps is not None else []
        self._prototype_columns: list[str] | None = None
        if data is not None:
            info = SchemaInfo.from_frame(_frame.from_native(data))
            self._prototype_columns = info.columns
        self.roles = Roles()
        if formula is not None:
            if self._prototype_columns is None:
                raise ValueError("formula requires a prototype `data=` frame")
            self.roles = parse_formula(formula, self._prototype_columns)

    def _clone(self) -> "Recipe":
        """A shallow copy that preserves steps, prototype, and roles."""
        new = Recipe(steps=self.steps)
        new._prototype_columns = (
            list(self._prototype_columns)
            if self._prototype_columns is not None
            else None
        )
        new.roles = self.roles.copy()
        return new

    def add(self, *steps: Step) -> "Recipe":
        """Return a new recipe with ``steps`` appended (steps are pure)."""
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
        new = self._clone()
        assignments: dict[str, str | list[str] | None] = {
            "outcome": outcome,
            "predictor": predictor,
            "case_weights": case_weights,
            **custom,
        }
        for role, cols in assignments.items():
            if cols is None:
                continue
            for column in [cols] if isinstance(cols, str) else cols:
                self._check_column_exists(column)
                new.roles.add(column, role)
        return new

    def _check_column_exists(self, column: str) -> None:
        """Validate ``column`` against the prototype, if one was supplied."""
        if (
            self._prototype_columns is not None
            and column not in self._prototype_columns
        ):
            raise ValueError(f"column {column!r} is not in the prototype data")

    def fit(self, data: DataFrame) -> "FittedRecipe":
        """Fit each step in order on the training data.

        The transformed output of step *i* is the input to step *i+1*'s
        ``fit()``, so later steps see the schema as modified by earlier ones.
        ``RowOp`` steps are fitted but not applied here — row operations only
        affect training data and so never run during ``transform()`` either.
        """
        current = _frame.from_native(data)
        fitted_steps: list[Step] = []
        for step in self.steps:
            info = SchemaInfo.from_frame(current)
            fitted = step.fit(current, info)
            fitted.trained = True  # set by the recipe, not the step (per contract)
            if not isinstance(step, RowOp):
                current = fitted.transform(current)
            fitted_steps.append(fitted)
        return FittedRecipe(steps=fitted_steps, roles=self.roles.copy())


class FittedRecipe:
    """A fitted pipeline: applies stored statistics to new data."""

    def __init__(self, steps: list[Step], roles: Roles | None = None):
        self.steps = steps
        self.roles = roles if roles is not None else Roles()

    def transform(self, data: DataFrame) -> DataFrame:
        """Apply each fitted step in order, skipping ``RowOp`` steps.

        Never re-estimates. Returns a native frame at the sklearn boundary.
        """
        current = _frame.from_native(data)
        for step in self.steps:
            if isinstance(step, RowOp):
                continue
            current = step.transform(current)
        return _frame.to_native(current)
