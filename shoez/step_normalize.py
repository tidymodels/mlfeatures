"""Center-and-scale normalization: fit per-column mean/std, then standardize.

The *fit* half lives in :func:`fit_normalize` and the *apply* half in
:func:`apply_normalize`, both plain functions so the data-ops core can be
exercised on its own. :class:`StepNormalize` wraps them in the
:class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can fit
and apply it.

They are the pure ``prep`` / ``bake`` counterparts of R ``recipes``'
``step_normalize``: the statistics are estimated once at fit time and reused
unchanged at transform time as ``(x - mean) / std``.
"""

from __future__ import annotations

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
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_normalize", "apply_normalize", "StepNormalize"]


def fit_normalize(
    features: Frame,
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
        std = as_float(features[col].std(ddof=1))
        if std == 0.0:
            constant.append(col)
            continue
        columns.append(col)
        means_list.append(as_float(features[col].mean()))
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
    features: Frame,
    columns: list[str],
    means: npt.NDArray[np.float64],
    stds: npt.NDArray[np.float64],
) -> Frame:
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


class StepNormalize(Step):
    """Center and scale numeric columns to zero mean and unit variance.

    A ``ONE_TO_ONE`` step: each selected numeric column ``x`` is replaced in
    place by ``(x - mean) / std``, with the mean and standard deviation
    estimated at fit time. The statistical work is done by :func:`fit_normalize`
    and :func:`apply_normalize`; this class wraps them in the
    :class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can
    fit and apply it.

    Constant (zero-variance) columns among the selection cannot be scaled to
    unit variance; :func:`fit_normalize` drops them from the normalization (with
    a warning), and they pass through ``transform`` untouched.

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
        self.id = id if id is not None else "normalize"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: the normalizable columns and their statistics.
        self._columns: list[str] | None = None
        self._means: npt.NDArray[np.float64] | None = None
        self._stds: npt.NDArray[np.float64] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate per-column mean and std over the selected columns."""
        features = _frame.select(training_data, self.columns, info.roles)
        self._columns, self._means, self._stds = fit_normalize(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply the stored mean and std: ``(x - mean) / std``. Never re-estimates."""
        if self._columns is None or self._means is None or self._stds is None:
            raise RuntimeError("StepNormalize.transform() called before fit().")
        return apply_normalize(new_data, self._columns, self._means, self._stds)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepNormalize.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, normalized ones now float."""
        if self._schema_in is None or self._columns is None:
            raise RuntimeError("StepNormalize.schema_out() called before fit().")
        normalized = set(self._columns)
        dtypes = {
            col: ("Float64" if col in normalized else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Normalization is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Centering shifts zeros off zero, but output stays dense."""
        return False
