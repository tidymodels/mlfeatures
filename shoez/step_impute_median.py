"""Impute missing numeric values with the column median.

The *fit* half lives in `fit_impute_median()` and the *apply* half in
`apply_impute_median()`, both plain functions so the data-ops core can
be exercised on its own. `StepImputeMedian` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

A ``ONE_TO_ONE`` step (see ``design/steps-catalog.md``): each selected numeric
column has its null values filled with that column's median, estimated at fit
time and reused unchanged at transform time. Ports R ``recipes``'
``step_impute_median``. The median is robust to outliers, so it is often
preferred over the mean for skewed columns.

Because the median is a float and ``fill_null`` unifies dtypes, imputed columns
are returned as ``Float64`` (an integer column with imputed values is promoted).
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez._frame import as_float
from shoez.step import Step
from shoez.types import (
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
    features: Frame,
) -> dict[str, float]:
    """Estimate the per-column median used to impute missing values.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Returns
    -------
    ``dict`` mapping every input column name to its median. Empty when
    ``features`` has no columns.

    A column whose values are *all* null has an undefined (``NaN``) median and
    so cannot be imputed. It is **not** dropped: it stays in the mapping with a
    ``NaN`` value, `apply_impute_median()` leaves it unchanged, and a
    `UserWarning` is emitted naming such columns.
    """
    medians = {col: as_float(features[col].median()) for col in features.columns}

    all_null = [col for col, med in medians.items() if math.isnan(med)]
    if all_null:
        warnings.warn(
            f"{len(all_null)} column(s) are entirely null and cannot be "
            f"median-imputed; left unchanged: {all_null!r}",
            stacklevel=2,
        )

    return medians


def apply_impute_median(
    features: Frame,
    medians: dict[str, float],
) -> Frame:
    """Fill nulls with the stored per-column median.

    Parameters
    ----------
    features:
        A narwhals DataFrame to impute.
    medians:
        The ``{column: median}`` mapping returned by `fit_impute_median()`. Any
        column of ``features`` not present as a key is passed through untouched.
        A ``NaN`` median marks a column that could not be estimated (entirely
        null at fit time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame — the input frame with nulls in each mapped column filled
    by its median (columns with a ``NaN`` median are left as-is). Column names,
    order, and structure are unchanged.
    """
    # Skip columns with a NaN median (entirely null at fit time): leave as-is.
    exprs = [
        nw.col(col).fill_null(med).alias(col)
        for col, med in medians.items()
        if not math.isnan(med)
    ]
    return features.with_columns(*exprs)


class StepImputeMedian(Step):
    """Impute missing numeric values with the column median.

    A ``ONE_TO_ONE`` step: each selected numeric column has its nulls filled in
    place with that column's median, estimated at fit time. The work is done by
    `fit_impute_median()` and `apply_impute_median()`; this class
    wraps them in the `Step` lifecycle so a
    `Recipe` can fit and apply it.

    A column that is entirely null at fit time has no median to impute with; it
    is left unchanged (not dropped) with a warning, and passes through
    ``transform`` untouched.

    Examples
    --------
    The null is filled in place with the column median of the observed values:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepImputeMedian
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepImputeMedian(s.numeric_predictors()))
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

    def __init__(
        self,
        columns: Selector,
        id: str | None = None,
    ):
        self.columns = columns
        self.id = id if id is not None else "impute_median"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: {column: median} over the imputable columns.
        self._estimates: dict[str, float] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column median over the selected columns."""
        features = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_impute_median(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
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
        # Only columns that were actually imputed (non-NaN median) become float;
        # an entirely-null column is left unchanged and keeps its input dtype.
        imputed = {col for col, med in self._estimates.items() if not math.isnan(med)}
        dtypes = {
            col: ("Float64" if col in imputed else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Median imputation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_median`` never produces sparse output."""
        return False
