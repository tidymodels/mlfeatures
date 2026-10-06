"""Shared fit/apply/schema logic for `StepImputeMean` and `StepImputeMedian`.

Both steps fill nulls with a single per-column statistic (mean or median) and
share every behaviour that follows from that shape: a column whose values are
entirely null has an undefined (``NaN``) statistic and is left unchanged with a
warning rather than dropped, and an imputed column is promoted to ``Float64``.
This module holds that shared logic, parameterized by the reducer
(``Series.mean`` / ``Series.median``) and the label used in the warning text;
each step module keeps its own public `fit_impute_mean()` / `fit_impute_median()`
etc. with their own docstrings, delegating to the functions here.

Not part of the public API: only `shoez.step_impute_mean` and
`shoez.step_impute_median` import from here.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable

import narwhals.stable.v2 as nw

from shoez.types import ColSchema, Frame


def fit_reduce_impute(
    selected: Frame,
    reducer: Callable[[nw.Series[Any]], float],
    label: str,
) -> dict[str, float]:
    """Estimate the per-column statistic ``reducer`` used to impute nulls.

    A column whose values are entirely null has an undefined (``NaN``)
    statistic and so cannot be imputed; it stays in the returned mapping with
    a ``NaN`` value, and a ``UserWarning`` naming such columns is emitted with
    ``label`` (e.g. ``"mean"``, ``"median"``) describing what could not be
    computed.
    """
    values = {col: reducer(selected[col]) for col in selected.columns}

    all_null = [col for col, v in values.items() if math.isnan(v)]
    if all_null:
        warnings.warn(
            f"{len(all_null)} column(s) are entirely null and cannot be "
            f"{label}-imputed; left unchanged: {all_null!r}",
            stacklevel=3,
        )

    return values


def apply_reduce_impute(selected: Frame, values: dict[str, float]) -> Frame:
    """Fill nulls in ``selected`` with the stored per-column ``values``.

    A ``NaN`` value marks a column that could not be estimated (entirely null
    at fit time); such columns are left unchanged.
    """
    exprs = [
        nw.col(col).fill_null(v).alias(col)
        for col, v in values.items()
        if not math.isnan(v)
    ]
    return selected.with_columns(*exprs)


def reduce_impute_schema_out(
    schema_in: ColSchema, estimates: dict[str, float]
) -> ColSchema:
    """``schema_out()`` after reducer-based imputation: imputed columns become Float64.

    Only the columns that were actually imputed (a non-``NaN`` estimate)
    become ``Float64``; an entirely-null column is left unchanged and keeps
    its input dtype.
    """
    imputed = {col for col, v in estimates.items() if not math.isnan(v)}
    dtypes = {
        col: ("Float64" if col in imputed else schema_in.dtypes[col])
        for col in schema_in.columns
    }
    return ColSchema(columns=schema_in.columns, dtypes=dtypes)
