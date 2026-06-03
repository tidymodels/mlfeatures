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

from typing import Literal

from shoez import _frame
from shoez.step import RowOp, Step
from shoez.step_corr import StepCorr
from shoez.types import DataFrame, SchemaInfo

__all__ = ["Recipe", "FittedRecipe"]


class Recipe:
    """A declarative, unfitted pipeline of feature-engineering steps."""

    def __init__(self, steps: list[Step] | None = None):
        self.steps: list[Step] = list(steps) if steps is not None else []

    def _append(self, step: Step) -> "Recipe":
        """Return a new recipe with ``step`` appended (steps are pure)."""
        return Recipe(steps=[*self.steps, step])

    def step_corr(
        self,
        columns: list[str],
        threshold: float = 0.9,
        method: Literal["pearson", "spearman"] = "pearson",
        id: str | None = None,
    ) -> "Recipe":
        """Append a :class:`~shoez.step_corr.StepCorr` and return a new recipe."""
        return self._append(
            StepCorr(columns, threshold=threshold, method=method, id=id)
        )

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
        return FittedRecipe(steps=fitted_steps)


class FittedRecipe:
    """A fitted pipeline: applies stored statistics to new data."""

    def __init__(self, steps: list[Step]):
        self.steps = steps

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
