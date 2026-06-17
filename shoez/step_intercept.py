"""Intercept: add a constant column.

The *apply* half lives in `add_intercept()` as a plain function so the logic
can be exercised on its own; `StepIntercept` wraps it in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

This is the ``ADDING``-mode prototype from ``design/steps-catalog.md``, based on
``step_intercept`` from the R ``recipes`` package and generalised from a column
of ones to a configurable fixed value. An intercept estimates nothing from the
data, so unlike a filter such as `fit_drop_corr()`
there is no *fit* half — only this *apply* half.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals as nw

from shoez.step import Step
from shoez.types import (
    ANY_DTYPE,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["add_intercept", "StepIntercept"]


def add_intercept(
    features: Frame, value: float = 1.0, name: str = "intercept"
) -> Frame:
    """Append a constant column of ``value`` named ``name``.

    Parameters
    ----------
    features
        Internal `Frame` (narwhals). Native frames are
        wrapped via `shoez._frame.from_native()` at the recipe boundary.
    value
        The constant to fill the new column with. Default ``1.0`` — a column of
        ones, the classic regression intercept.
    name
        Name of the new column. Default ``"intercept"``.

    Returns
    -------
    Frame
        ``features`` with one extra column appended. The literal broadcasts
        across every row, so 0-row and 1-row frames still produce a
        schema-correct constant column.

    A clash with an existing column is not checked here: within a recipe the
    framework rejects it up front (``Schema.check_additions``), the single
    centralized name-collision guard for every column-adding step.
    """
    return features.with_columns(nw.lit(value).alias(name))


class StepIntercept(Step):
    """Add a constant column with a fixed value.

    An ``ADDING`` step: it consumes no input columns and appends a single
    constant column (default a column of ones named ``"intercept"``). The column
    is produced by `add_intercept()`; this class wraps it in the
    `Step` lifecycle so a `Recipe` can
    fit and apply it.

    Because it adds rather than transforms existing columns, there is nothing to
    estimate: ``fit()`` only records the produced schema.
    """

    mode = Mode.ADDING

    def __init__(
        self,
        value: float = 1.0,
        name: str = "intercept",
        role: str | None = "predictor",
        id: str | None = None,
    ):
        self.value = value
        self.name = name
        self.id = id if id is not None else "intercept"
        self.trained = False
        self.keep_original_cols = True  # ADDING never drops the source columns.
        self.output_role = role
        # Populated by ``fit()``.
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record the produced column's schema. Estimates nothing."""
        # Build the column on the training frame to read its concrete dtype.
        # ``add_intercept`` is idempotent, so the recipe re-applying it in
        # ``transform`` is harmless. A name clash is rejected by the recipe via
        # Schema.check_additions before this step runs.
        produced = add_intercept(training_data, self.value, self.name)
        out_dtype = str(produced.schema[self.name])
        self._schema_out = ColSchema(columns=[self.name], dtypes={self.name: out_dtype})
        # A constant column descends from no input column.
        self._output_parents = {self.name: []}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Append the constant column. Never re-estimates."""
        if self._schema_out is None:
            raise RuntimeError("StepIntercept.transform() called before fit().")
        return add_intercept(new_data, self.value, self.name)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed — none; it only adds."""
        return ColSchema(columns=[])

    def schema_out(self) -> ColSchema:
        """Schema of the single column this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepIntercept.schema_out() called before fit().")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """The intercept column is a constant — it has no parent columns."""
        if self._output_parents is None:
            raise RuntimeError("StepIntercept.output_parents() called before fit().")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """No input columns are consumed, so no dtype constraint applies."""
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        """``step_intercept`` never produces sparse output."""
        return False
