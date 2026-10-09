"""Impute missing numeric values with the column mean.

The *fit* half lives in `fit_impute_mean()` and the *apply* half in
`apply_impute_mean()`, both plain functions so the data-ops core can be
exercised on its own. `StepImputeMean` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

Each selected numeric column has its null values filled with that column's
mean. Ports R
``recipes``' ``step_impute_mean``.

Because the mean is a float and ``fill_null`` unifies dtypes, imputed columns
are returned as ``Float64`` (an integer column with imputed values is promoted).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlfeatures import _frame
from mlfeatures._frame import as_float
from mlfeatures._impute_reduce import (
    apply_reduce_impute,
    fit_reduce_impute,
    reduce_impute_schema_out,
)
from mlfeatures.step import Step
from mlfeatures.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_impute_mean", "apply_impute_mean", "StepImputeMean"]


def fit_impute_mean(
    selected: Frame,
) -> dict[str, float]:
    """Estimate the per-column mean used to impute missing values.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Returns
    -------
    ``dict`` mapping every input column name to its mean. Empty when
    ``selected`` has no columns.

    A column whose values are *all* null has an undefined (``NaN``) mean and so
    cannot be imputed. It is **not** dropped: it stays in the mapping with a
    ``NaN`` value, `apply_impute_mean()` leaves it unchanged, and a
    `UserWarning` is emitted naming such columns.
    """
    return fit_reduce_impute(selected, lambda s: as_float(s.mean()), "mean")


def apply_impute_mean(
    selected: Frame,
    means: dict[str, float],
) -> Frame:
    """Fill nulls with the stored per-column mean.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to impute.
    means:
        The ``{column: mean}`` mapping returned by `fit_impute_mean()`. Any
        column of ``selected`` not present as a key is passed through untouched.
        A ``NaN`` mean marks a column that could not be estimated (entirely null
        at fit time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame — the input frame with nulls in each mapped column filled
    by its mean (columns with a ``NaN`` mean are left as-is). Column names,
    order, and structure are unchanged.
    """
    return apply_reduce_impute(selected, means)


class StepImputeMean(Step):
    """Impute missing numeric values with the column mean.

    Each selected numeric column has its nulls filled in place with that
    column's mean.

    A column that is entirely null at fit time has no mean to impute with; it is
    left unchanged (not dropped) with a warning, and passes through ``transform``
    untouched.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to impute. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    id
        Stable step identifier. Defaults to ``"impute_mean"``; a second step
        of this type in the same recipe becomes ``"impute_mean_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    The null is filled in place with the column mean of the observed values:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepImputeMean
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepImputeMean(s.numeric_features()))
    recipe
    ```

    `fit()` computes the mean of the observed values:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` fills the nulls in place:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    #: Populated by ``fit()``: {column: mean} over the imputable columns.
    _estimates: dict[str, float] | None

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        self.columns = columns
        super().__init__(id, "impute_mean")

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column mean over the selected columns."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_impute_mean(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Fill nulls with the stored means. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepImputeMean.transform() called before fit()")
        return apply_impute_mean(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMean.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, imputed ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepImputeMean.schema_out() called before fit()")
        return reduce_impute_schema_out(self._schema_in, self._estimates)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Mean imputation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_mean`` never produces sparse output."""
        return False
