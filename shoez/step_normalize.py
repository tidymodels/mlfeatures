"""Fitting and transform logic for the normalization (center-and-scale) step.

``fit_normalize`` estimates the per-column mean and standard deviation used to
standardize numeric columns to zero mean and unit variance, and
``apply_normalize`` applies those stored statistics as ``(x - mean) / std``.
They are the pure ``prep`` / ``bake`` counterparts of R ``recipes``'
``step_normalize``: the statistics are estimated once at fit time and reused
unchanged at transform time.
"""

from __future__ import annotations

import warnings
from typing import Any

import narwhals as nw
import numpy as np
import numpy.typing as npt

__all__ = ["fit_normalize", "apply_normalize"]


def _as_float(value: Any) -> float:
    """Coerce a narwhals scalar to ``float``, mapping a missing value to NaN."""
    return float("nan") if value is None else float(value)


def fit_normalize(
    features: nw.DataFrame[Any],
) -> tuple[list[str], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate per-column mean and standard deviation for normalization.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Callers are expected to pass an all-numeric frame; dtype screening is the
    selector / ``accepts_dtypes`` layer's job, so this function does not
    re-validate it.

    Returns
    -------
    tuple ``(columns, means, stds)`` — ``columns`` is the list of column names
    that were normalizable, and ``means`` / ``stds`` are ``numpy.float64``
    arrays parallel to ``columns`` (``means[i]`` / ``stds[i]`` belong to
    ``columns[i]``).  All three are empty when ``features`` has no columns.

    Columns with a standard deviation of ``0`` (constant columns — every value
    the same) cannot be scaled to unit variance, so they are **omitted** from
    the result and a :class:`UserWarning` is emitted naming them.

    Notes
    -----
    Based on the ``prep`` step of ``step_normalize`` in the R ``recipes``
    package.  Uses the sample standard deviation (``ddof=1``, pinned explicitly),
    matching R's ``sd()``; scaling by it makes the normalized column's sample
    standard deviation exactly ``1``.  A ``NaN`` standard deviation (e.g. from a
    single-row fit) is not treated as constant — it is kept and propagates as
    ``NaN`` at transform time, signalling a degenerate fit.
    """
    columns: list[str] = []
    means_list: list[float] = []
    stds_list: list[float] = []
    constant: list[str] = []

    for col in features.columns:
        # ddof=1 (sample std) explicitly: scaling by it makes the sample std of
        # the normalized column exactly 1, matching R's sd().
        std = _as_float(features[col].std(ddof=1))
        if std == 0.0:
            constant.append(col)
            continue
        columns.append(col)
        means_list.append(_as_float(features[col].mean()))
        stds_list.append(std)

    if constant:
        warnings.warn(
            f"{len(constant)} column(s) had zero standard deviation (constant) "
            f"and were not normalized: {constant!r}.",
            stacklevel=2,
        )

    means = np.asarray(means_list, dtype=np.float64)
    stds = np.asarray(stds_list, dtype=np.float64)
    return columns, means, stds


def apply_normalize(
    features: nw.DataFrame[Any],
    columns: list[str],
    means: npt.NDArray[np.float64],
    stds: npt.NDArray[np.float64],
) -> nw.DataFrame[Any]:
    """Standardize ``features`` as ``(x - mean) / std`` using stored statistics.

    Parameters
    ----------
    features:
        A narwhals DataFrame to normalize.
    columns:
        The column names to normalize — the ``columns`` returned by
        :func:`fit_normalize`. Any column of ``features`` not in this list (e.g.
        a constant column dropped at fit time) is passed through untouched.
    means:
        Per-column means from :func:`fit_normalize`, parallel to ``columns``.
    stds:
        Per-column standard deviations from :func:`fit_normalize`, parallel to
        ``columns``.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``columns`` replaced
    in place by its standardized value. Column names, order, and structure are
    unchanged.

    Raises
    ------
    ValueError
        If ``means`` or ``stds`` does not have exactly one entry per name in
        ``columns``.
    """
    if len(means) != len(columns) or len(stds) != len(columns):
        raise ValueError(
            f"`means` ({len(means)}) and `stds` ({len(stds)}) must each have one "
            f"entry per name in `columns` ({len(columns)})."
        )

    exprs = [
        ((nw.col(col) - float(means[i])) / float(stds[i])).alias(col)
        for i, col in enumerate(columns)
    ]
    return features.with_columns(*exprs)
