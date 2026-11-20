"""Impute missing numeric values with the column median.

The *fit* half lives in `fit_impute_median()` and the *apply* half in
`apply_impute_median()`, both plain functions so the data-ops core can
be exercised on its own. `StepImputeMedian` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

Each selected numeric column has its null values filled with that column's
median. Ports R
``recipes``' ``step_impute_median``. The median is robust to outliers, so it is
often preferred over the mean for skewed columns.

Because the median is a float and ``fill_null`` unifies dtypes, imputed columns
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

__all__ = [
    "fit_impute_median",
    "apply_impute_median",
    "StepImputeMedian",
]


def fit_impute_median(
    selected: Frame,
) -> dict[str, float]:
    """Estimate the per-column median used to impute missing values.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Returns
    -------
    ``dict`` mapping every input column name to its median. Empty when
    ``selected`` has no columns.

    A column whose values are *all* null has an undefined (``NaN``) median and
    so cannot be imputed. It is **not** dropped: it stays in the mapping with a
    ``NaN`` value, `apply_impute_median()` leaves it unchanged, and a
    `UserWarning` is emitted naming such columns.
    """
    return fit_reduce_impute(selected, lambda s: as_float(s.median()), "median")


def apply_impute_median(
    selected: Frame,
    medians: dict[str, float],
) -> Frame:
    """Fill nulls with the stored per-column median.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to impute.
    medians:
        The ``{column: median}`` mapping returned by `fit_impute_median()`. Any
        column of ``selected`` not present as a key is passed through untouched.
        A ``NaN`` median marks a column that could not be estimated (entirely
        null at fit time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame: the input data frame with nulls in each mapped column filled
    by its median (columns with a ``NaN`` median are left as-is). Column names,
    order, and structure are unchanged.
    """
    return apply_reduce_impute(selected, medians)


class StepImputeMedian(Step):
    """Impute missing numeric values with the column median.

    Each selected numeric column has its nulls filled in place with that
    column's median, which is less sensitive to outliers than the mean (see
    `StepImputeMean`). Categorical columns need `StepImputeMode`. To keep a
    record of which values were missing, see `StepIndicateNa`.

    A column that is entirely null at fit time has no median to impute with; it
    is left unchanged (not dropped) with a warning, and passes through
    ``transform`` untouched.

    See [Simple Imputation](https://feaz-book.com/missing-simple) in *Feature
    Engineering A to Z* for background on this technique.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to impute. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the data frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    id
        Stable step identifier. Defaults to ``"impute_median"``; a second step
        of this type in the same recipe becomes ``"impute_median_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    The null is filled in place with the column median of the observed values:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepImputeMedian
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepImputeMedian(s.numeric_features()))
    recipe
    ```

    `fit()` computes the median of the observed values:

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

    #: Populated by ``fit()``: {column: median} over the imputable columns.
    _estimates: dict[str, float] | None

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        self.columns = columns
        super().__init__(id, "impute_median")

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column median over the selected columns."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_impute_median(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Fill nulls with the stored medians. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepImputeMedian.transform() called before fit()")
        return apply_impute_median(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMedian.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, imputed ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepImputeMedian.schema_out() called before fit()")
        return reduce_impute_schema_out(self._schema_in, self._estimates)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Median imputation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_median`` never produces sparse output."""
        return False
