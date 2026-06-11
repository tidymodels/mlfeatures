"""Fitting logic for the correlation-drop step.

``fit_drop_corr`` identifies which numeric columns should be dropped because
their pairwise absolute correlation exceeds a threshold.  Zero-variance
(all-NaN correlation) columns are dropped rather than silently excluded.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import narwhals as nw
import numpy as np

from shoez import _frame
from shoez.step import Step
from shoez.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_drop_corr", "StepDropCorr"]


def fit_drop_corr(
    features: nw.DataFrame[Any],
    threshold: float = 0.9,
    method: Literal["pearson", "spearman"] = "pearson",
) -> list[str]:
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
    list[str] — column names to drop.  Empty list if nothing exceeds the threshold.

    Notes
    -----
    Uses pairwise-complete observations via ``nw.corr()``, which delegates NaN
    handling to the underlying backend.  When two columns in a correlated pair
    must be chosen for removal, the one with the higher mean absolute
    correlation across all columns is dropped; ties go to the row variable.

    Zero-variance / all-NaN columns are added to the drop list rather than
    excluded from the filter but kept.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"`threshold` must be between 0 and 1, got {threshold!r}.")
    if method not in ("pearson", "spearman"):
        raise ValueError(f"`method` must be 'pearson' or 'spearman', got {method!r}.")

    col_names = features.columns
    n = len(col_names)

    if n < 2 or features.shape[0] == 0:
        return []

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

    # Drop all-NA columns rather than keeping them.
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
        return na_drop

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

    # Dense rank.
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
    return list(dict.fromkeys(na_drop + corr_drop_names))


class StepDropCorr(Step):
    """Remove numeric columns whose pairwise absolute correlation is too high.

    A ``REMOVING`` step: the selected columns that exceed ``threshold`` are
    dropped, and no new columns are produced. The statistical decision is made by
    `fit_drop_corr()`; this class wraps it in the `Step`
    lifecycle so a `Recipe` can fit and apply it.

    The columns to consider may be given as a single name, an explicit list of
    names, a narwhals selector (e.g. ``narwhals.selectors.numeric()``), or a
    polars selector (``polars.selectors``). Any selector is resolved against the
    frame's schema at fit time, regardless of backend. See
    ``design/selectors.md``.
    """

    mode = Mode.REMOVING

    def __init__(
        self,
        columns: Selector,
        threshold: float = 0.9,
        method: Literal["pearson", "spearman"] = "pearson",
        id: str | None = None,
    ):
        # Held verbatim. Interpreting a column spec (name, list, or selector) is
        # selection logic that belongs to the frame boundary, not the step:
        # ``fit()`` hands it to ``_frame.select``, which resolves it at fit time.
        self.columns = columns
        self.threshold = threshold
        self.method: Literal["pearson", "spearman"] = method
        self.id = id if id is not None else "corr"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``.
        self._to_drop: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate which selected columns to drop and store the decision."""
        features = _frame.select(training_data, self.columns, info.roles)
        self._to_drop = fit_drop_corr(features, self.threshold, self.method)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        # Record the resolved column names (a selector is concrete only now).
        self._schema_in = ColSchema(columns=list(features.columns), dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Drop the columns chosen at fit time. Never re-estimates."""
        if self._to_drop is None:
            raise RuntimeError("StepDropCorr.transform() called before fit().")
        return _frame.drop(new_data, self._to_drop)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDropCorr.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the columns this step produced — none; it only removes."""
        if self._to_drop is None:
            raise RuntimeError("StepDropCorr.schema_out() called before fit().")
        return ColSchema(columns=[])

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Correlation is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._to_drop:
            for col in self._to_drop:
                for role in list(roles.of(col)):
                    roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """``step_drop_corr`` never produces sparse output."""
        return False
