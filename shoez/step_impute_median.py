"""Impute missing numeric values with the column median.

The *fit* half lives in :func:`fit_impute_median` and the *apply* half in
:func:`apply_impute_median`, both plain functions so the data-ops core can
be exercised on its own. :class:`StepImputeMedian` wraps them in the
:class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can fit
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
import numpy as np
import numpy.typing as npt

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
) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Estimate the per-column median used to impute missing values.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Returns
    -------
    tuple ``(columns, medians)`` — ``columns`` is every input column name and
    ``medians`` is a ``numpy.float64`` array parallel to it (``medians[i]`` is
    the median of ``columns[i]``). Both are empty when ``features`` has no
    columns.

    A column whose values are *all* null has an undefined (``NaN``) median and
    so cannot be imputed. It is **not** dropped: it stays in ``columns`` with a
    ``NaN`` median, :func:`apply_impute_median` leaves it unchanged, and a
    :class:`UserWarning` is emitted naming such columns.
    """
    columns = list(features.columns)
    medians = [as_float(features[col].median()) for col in columns]

    all_null = [col for col, med in zip(columns, medians) if math.isnan(med)]
    if all_null:
        warnings.warn(
            f"{len(all_null)} column(s) are entirely null and cannot be "
            f"median-imputed; left unchanged: {all_null!r}.",
            stacklevel=2,
        )

    return columns, np.asarray(medians, dtype=np.float64)


def apply_impute_median(
    features: Frame,
    columns: list[str],
    medians: npt.NDArray[np.float64],
) -> Frame:
    """Fill nulls in ``columns`` with the stored per-column median.

    Parameters
    ----------
    features:
        A narwhals DataFrame to impute.
    columns:
        The column names to impute — the ``columns`` returned by
        :func:`fit_impute_median`. Any column of ``features`` not in this
        list is passed through untouched.
    medians:
        Per-column medians from :func:`fit_impute_median`, parallel to
        ``columns``. A ``NaN`` median marks a column that could not be estimated
        (entirely null at fit time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame — the input frame with nulls in each column of
    ``columns`` filled by its median (columns with a ``NaN`` median are left
    as-is). Column names, order, and structure are unchanged.

    Raises
    ------
    ValueError
        If ``medians`` does not have exactly one entry per name in ``columns``.
    """
    if len(medians) != len(columns):
        raise ValueError(
            f"`medians` ({len(medians)}) must have one entry per name in "
            f"`columns` ({len(columns)})."
        )

    # Skip columns with a NaN median (entirely null at fit time): leave as-is.
    exprs = [
        nw.col(col).fill_null(float(medians[i])).alias(col)
        for i, col in enumerate(columns)
        if not math.isnan(medians[i])
    ]
    return features.with_columns(*exprs)


class StepImputeMedian(Step):
    """Impute missing numeric values with the column median.

    A ``ONE_TO_ONE`` step: each selected numeric column has its nulls filled in
    place with that column's median, estimated at fit time. The work is done by
    :func:`fit_impute_median` and :func:`apply_impute_median`; this class
    wraps them in the :class:`~shoez.step.Step` lifecycle so a
    :class:`~shoez.recipe.Recipe` can fit and apply it.

    A column that is entirely null at fit time has no median to impute with; it
    is left unchanged (not dropped) with a warning, and passes through
    ``transform`` untouched.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: list[str],
        id: str | None = None,
    ):
        self.columns = list(columns)
        self.id = id if id is not None else "impute_median"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: the columns and their medians.
        self._columns: list[str] | None = None
        self._medians: npt.NDArray[np.float64] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column median over the selected columns."""
        features = _frame.select(training_data, self.columns)
        self._columns, self._medians = fit_impute_median(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=list(self.columns), dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Fill nulls with the stored medians. Never re-estimates."""
        if self._columns is None or self._medians is None:
            raise RuntimeError("StepImputeMedian.transform() called before fit().")
        return apply_impute_median(new_data, self._columns, self._medians)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMedian.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, imputed ones now float."""
        if self._schema_in is None or self._columns is None or self._medians is None:
            raise RuntimeError("StepImputeMedian.schema_out() called before fit().")
        # Only columns that were actually imputed (non-NaN median) become float;
        # an entirely-null column is left unchanged and keeps its input dtype.
        imputed = {
            col for col, med in zip(self._columns, self._medians) if not math.isnan(med)
        }
        dtypes = {
            col: ("Float64" if col in imputed else self._schema_in.dtypes[col])
            for col in self.columns
        }
        return ColSchema(columns=list(self.columns), dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Median imputation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_median`` never produces sparse output."""
        return False
