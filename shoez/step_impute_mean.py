"""Impute missing numeric values with the column mean.

The *fit* half lives in `fit_impute_mean()` and the *apply* half in
`apply_impute_mean()`, both plain functions so the data-ops core can be
exercised on its own. `StepImputeMean` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

A ``ONE_TO_ONE`` step (see ``design/steps-catalog.md``): each selected numeric
column has its null values filled with that column's mean, estimated at fit time
and reused unchanged at transform time. Ports R ``recipes``' ``step_impute_mean``.

Because the mean is a float and ``fill_null`` unifies dtypes, imputed columns
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

__all__ = ["fit_impute_mean", "apply_impute_mean", "StepImputeMean"]


def fit_impute_mean(
    features: Frame,
) -> dict[str, float]:
    """Estimate the per-column mean used to impute missing values.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Returns
    -------
    ``dict`` mapping every input column name to its mean. Empty when
    ``features`` has no columns.

    A column whose values are *all* null has an undefined (``NaN``) mean and so
    cannot be imputed. It is **not** dropped: it stays in the mapping with a
    ``NaN`` value, `apply_impute_mean()` leaves it unchanged, and a
    `UserWarning` is emitted naming such columns.
    """
    means = {col: as_float(features[col].mean()) for col in features.columns}

    all_null = [col for col, mean in means.items() if math.isnan(mean)]
    if all_null:
        warnings.warn(
            f"{len(all_null)} column(s) are entirely null and cannot be "
            f"mean-imputed; left unchanged: {all_null!r}.",
            stacklevel=2,
        )

    return means


def apply_impute_mean(
    features: Frame,
    means: dict[str, float],
) -> Frame:
    """Fill nulls with the stored per-column mean.

    Parameters
    ----------
    features:
        A narwhals DataFrame to impute.
    means:
        The ``{column: mean}`` mapping returned by `fit_impute_mean()`. Any
        column of ``features`` not present as a key is passed through untouched.
        A ``NaN`` mean marks a column that could not be estimated (entirely null
        at fit time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame — the input frame with nulls in each mapped column filled
    by its mean (columns with a ``NaN`` mean are left as-is). Column names,
    order, and structure are unchanged.
    """
    # Skip columns with a NaN mean (entirely null at fit time): leave them as-is.
    exprs = [
        nw.col(col).fill_null(mean).alias(col)
        for col, mean in means.items()
        if not math.isnan(mean)
    ]
    return features.with_columns(*exprs)


class StepImputeMean(Step):
    """Impute missing numeric values with the column mean.

    A ``ONE_TO_ONE`` step: each selected numeric column has its nulls filled in
    place with that column's mean, estimated at fit time. The work is done by
    `fit_impute_mean()` and `apply_impute_mean()`; this class
    wraps them in the `Step` lifecycle so a
    `Recipe` can fit and apply it.

    A column that is entirely null at fit time has no mean to impute with; it is
    left unchanged (not dropped) with a warning, and passes through ``transform``
    untouched.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        id: str | None = None,
    ):
        self.columns = columns
        self.id = id if id is not None else "impute_mean"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: {column: mean} over the imputable columns.
        self._estimates: dict[str, float] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column mean over the selected columns."""
        features = _frame.select(training_data, self.columns, info.roles)
        self._estimates = fit_impute_mean(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Fill nulls with the stored means. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepImputeMean.transform() called before fit().")
        return apply_impute_mean(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMean.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, imputed ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepImputeMean.schema_out() called before fit().")
        # Only columns that were actually imputed (non-NaN mean) become float;
        # an entirely-null column is left unchanged and keeps its input dtype.
        imputed = {col for col, mean in self._estimates.items() if not math.isnan(mean)}
        dtypes = {
            col: ("Float64" if col in imputed else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Mean imputation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_mean`` never produces sparse output."""
        return False
