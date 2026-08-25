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
    selected: Frame, value: float = 1.0, name: str = "intercept"
) -> Frame:
    """Append a constant column of ``value`` named ``name``.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals). Native frames are
        wrapped via `shoez._frame.from_native()` at the recipe boundary.
    value
        The constant to fill the new column with. Default ``1.0`` — a column of
        ones, the classic regression intercept. The column's dtype follows the
        literal, so an ``int`` ``value`` gives an integer column.
    name
        Name of the new column. Default ``"intercept"``.

    Returns
    -------
    Frame
        ``selected`` with one extra column appended. The literal broadcasts
        across every row, so 0-row and 1-row frames still produce a
        schema-correct constant column. An existing column of the same name is
        overwritten in place; note that reaching this function through
        `StepIntercept` in a `Recipe` cannot happen, since the recipe rejects a
        step that produces an already-present column name.
    """
    return selected.with_columns(nw.lit(value).alias(name))


class StepIntercept(Step):
    """Add a constant column with a fixed value.

    A single constant column is added, by default a column of ones named
    ``"intercept"``.

    Parameters
    ----------
    value
        The constant to fill the new column with. Default ``1.0``. The column's
        dtype follows the literal, so an ``int`` gives an integer column.
    name
        Name of the new column. Default ``"intercept"``. A `Recipe` rejects the
        step if the frame already has a column of this name.
    role
        Role assigned to the new column (see `Roles`). Default ``"feature"``.
    id
        Stable step identifier. Defaults to ``"intercept"``; a second step
        of this type in the same recipe becomes ``"intercept_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    A constant ``intercept`` column of ones is appended; the input columns pass
    through untouched:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepIntercept
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepIntercept())
    recipe
    ```

    `fit()` has nothing to estimate; the step is constant:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` appends the column:

    ```{python}
    fitted.transform(df)
    ```

    ``value`` and ``name`` control what is appended:

    ```{python}
    recipe = Recipe().add(StepIntercept(value=0.0, name="bias"))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ADDING

    def __init__(
        self,
        value: float = 1.0,
        name: str = "intercept",
        role: str | None = "feature",
        id: str | None = None,
    ):
        self.value = value
        self.name = name
        self._init_id(id, "intercept")
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
        # ``transform`` is harmless.
        produced = add_intercept(training_data, self.value, self.name)
        out_dtype = str(produced.schema[self.name])
        self._schema_out = ColSchema(columns=[self.name], dtypes={self.name: out_dtype})
        # A constant column descends from no input column.
        self._output_parents = {self.name: []}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Append the constant column. Never re-estimates."""
        if self._schema_out is None:
            raise RuntimeError("StepIntercept.transform() called before fit()")
        return add_intercept(new_data, self.value, self.name)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed — none; it only adds."""
        return ColSchema(columns=[])

    def schema_out(self) -> ColSchema:
        """Schema of the single column this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepIntercept.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """The intercept column is a constant — it has no parent columns."""
        if self._output_parents is None:
            raise RuntimeError("StepIntercept.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """No input columns are consumed, so no dtype constraint applies."""
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        """``step_intercept`` never produces sparse output."""
        return False
