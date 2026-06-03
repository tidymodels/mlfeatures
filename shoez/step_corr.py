"""Fitting logic for the correlation-filter step.

``fit_filter_corr`` identifies which numeric columns should be dropped because
their pairwise absolute correlation exceeds a threshold.  It is based on the
``corr_filter`` function from the R ``recipes`` package, with one deviation:
zero-variance (all-NaN correlation) columns are dropped rather than silently
excluded.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import narwhals as nw
import numpy as np
import numpy.typing as npt

__all__ = ["fit_filter_corr"]


def fit_filter_corr(
    features: nw.DataFrame[Any],
    threshold: float = 0.9,
    method: Literal["pearson", "spearman"] = "pearson",
) -> npt.NDArray[np.str_]:
    """Identify columns to remove due to high pairwise correlation.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.
    threshold:
        Pairs with |r| strictly above this value trigger removal. Default 0.9.
    method:
        Correlation method: ``"pearson"`` or ``"spearman"``.  Kendall is not
        supported by ``nw.corr()``.

    Returns
    -------
    numpy.ndarray of dtype ``numpy.str_`` — column names to drop.  Empty array
    if nothing exceeds the threshold.

    Notes
    -----
    Based on ``corr_filter`` from the R ``recipes`` package.  Uses
    pairwise-complete observations via ``nw.corr()``, which delegates NaN
    handling to the underlying backend (equivalent to R's
    ``use = "pairwise.complete.obs"``).  When two columns in a correlated pair
    must be chosen for removal, the one with the higher mean absolute
    correlation across all columns is dropped; ties go to the row variable
    (matching R's behaviour).

    Deviation from R: zero-variance / all-NaN columns are added to the drop
    list rather than excluded from the filter but kept.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be between 0 and 1, got {threshold!r}.")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"`method` must be 'pearson' or 'spearman', got {method!r}.")

    col_names = features.columns
    n = len(col_names)

    # np.asarray on an empty list gives a float64 array by default; force
    # the dtype to unicode-string so the result type is stable.
    if n < 2 or features.shape[0] == 0:
        return np.asarray([], dtype=np.str_)

    # Build symmetric correlation matrix via pairwise nw.corr() calls.
    # nw.corr() uses pairwise-complete observations through the backend.
    corr = np.full((n, n), np.nan)
    np.fill_diagonal(corr, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            val = features.select(
                nw.corr(col_names[i], col_names[j], method=method)
            ).item()
            corr[i, j] = float(val)
            corr[j, i] = float(val)

    # Detect all-NA columns: every off-diagonal entry is NaN.
    # These arise from zero-variance or all-null input columns.
    off_diag = ~np.eye(n, dtype=bool)
    all_na = np.all(np.isnan(corr) | ~off_diag, axis=0)

    # Deviation from R: drop all-NA columns rather than keeping them.
    na_drop: list[str] = [col_names[i] for i in range(n) if all_na[i]]
    if na_drop:
        warnings.warn(
            f"The correlation matrix has missing values. "
            f"{len(na_drop)} column(s) were removed from the data.",
            stacklevel=2,
        )
        corr[all_na, :] = 0.0
        corr[:, all_na] = 0.0
        np.fill_diagonal(corr, 1.0)

    # If fewer than 2 non-NA columns remain, skip the correlation filter.
    n_remaining = int((~all_na).sum())
    if n_remaining < 2:
        warnings.warn(
            "Too many correlations are NA; skipping correlation filter.",
            stacklevel=2,
        )
        return np.asarray(na_drop, dtype=np.str_)

    # Zero out sporadic NaN (pairs where one column had partial nulls).
    if np.any(np.isnan(corr)):
        warnings.warn(
            "The correlation matrix has sporadic missing values. "
            "Some columns were excluded from the filter.",
            stacklevel=2,
        )
        corr[np.isnan(corr)] = 0.0
        np.fill_diagonal(corr, 1.0)

    # Average absolute correlation per column — used to rank columns so the
    # one with the broadest correlation to others is removed first.
    avg_corr = np.mean(np.abs(corr), axis=0)

    # Dense rank (matches R's as.numeric(as.factor(averageCorr))).
    _, inv = np.unique(avg_corr, return_inverse=True)
    ranks = inv + 1  # 1-based

    # Consider only the upper triangle (each pair counted once).
    upper = np.abs(corr).astype(float)
    upper[np.tril_indices(n)] = np.nan

    rows_idx, cols_idx = np.where(upper > threshold)

    corr_drop: set[int] = set()
    for r, c in zip(rows_idx, cols_idx):
        if ranks[c] > ranks[r]:
            corr_drop.add(int(c))
        else:
            corr_drop.add(int(r))

    corr_drop_names = [col_names[i] for i in sorted(corr_drop)]

    # Combine zero-variance drops and correlation-based drops, preserving
    # original column order and deduplicating.
    all_drop = dict.fromkeys(na_drop + corr_drop_names)
    result = list(all_drop.keys())

    # np.asarray on an empty list gives a float64 array by default; force
    # the dtype to unicode-string so the result type is stable.
    return np.asarray(result, dtype=np.str_)
