"""Ordered-quantile (ORQ) normalization: fit a per-column rank-to-normal map, apply it.

The *fit* half lives in `fit_order_norm()` and the *apply* half in
`apply_order_norm()`, both plain functions so the data-ops core can be
exercised on its own. `StepOrderNorm` wraps them in the
`Step` lifecycle so a `Recipe` can fit and apply it.

Port of ``orderNorm()`` / ``predict.orderNorm()`` from the R ``bestNormalize``
package, not a base R ``recipes`` step. Each selected column's training values
are ranked (ties averaged), the ranks mapped to normal quantiles via the Hazen
formula, and the resulting ``(x, z)`` pairs stored as a lookup table.
``transform()`` maps new values onto the same table by linear interpolation,
extrapolating past the training range using the boundary segment's slope
(``bestNormalize`` instead fits an auxiliary logistic regression for that
case; see `apply_order_norm` for the simplification made here).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np

from shoez import _frame
from shoez._dependencies import import_optional
from shoez._numeric_one_to_one import schema_out_for_touched
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

__all__ = ["OrderNormFit", "fit_order_norm", "apply_order_norm", "StepOrderNorm"]


@dataclass(frozen=True)
class OrderNormFit:
    """The rank-to-normal-quantile lookup table learned for one column.

    Attributes
    ----------
    x
        The column's distinct training values, sorted ascending.
    z
        The corresponding normal quantile for each value in ``x``, same length
        and order.
    """

    x: "np.ndarray[Any, Any]"
    z: "np.ndarray[Any, Any]"


def _average_ranks(x: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any]":
    """Ranks of ``x`` (1-based), with tied values receiving their average rank.

    Matches R's ``rank(x, ties.method = "average")``, which is what
    ``bestNormalize::orderNorm`` uses to turn values into ranks before mapping
    them to normal quantiles.
    """
    order = np.argsort(x, kind="stable")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    # 1-based positions of each element in sorted order.
    positions = np.arange(1, n + 1, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = positions[i : j + 1].mean()
        i = j + 1
    return ranks


def fit_order_norm(selected: Frame) -> dict[str, OrderNormFit]:
    """Learn a rank-to-normal-quantile lookup table per column.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Callers are expected to pass an all-numeric frame; dtype screening is the
    selector / ``accepts_dtypes`` layer's job, so this function does not
    re-validate it.

    Returns
    -------
    ``dict`` mapping each normalizable column name to its `OrderNormFit`.
    Empty when ``selected`` has no columns.

    Missing values are dropped before ranking. A column with fewer than 2
    distinct non-missing values (every value the same, or too few observations)
    cannot be mapped onto a spread of normal quantiles, so it is **omitted**
    from the result and a `UserWarning` is emitted naming it.

    Notes
    -----
    Based on ``orderNorm()`` in the R ``bestNormalize`` package: ranks are
    computed with ties averaged, then mapped to normal quantiles via the
    Hazen formula ``qnorm((rank - 0.5) / n)``, matching ``orderNorm()``'s
    in-range fitted values exactly. Out-of-range extrapolation differs from
    ``bestNormalize``, which fits an auxiliary logistic regression on ranks for
    that case; this implementation extrapolates linearly along the boundary
    segment's slope instead (see `apply_order_norm`), a deliberate
    simplification rather than a bug.
    """
    norm = import_optional("scipy.stats", purpose="by StepOrderNorm").norm

    fits: dict[str, OrderNormFit] = {}
    degenerate: list[str] = []

    for col in selected.columns:
        values = _frame.series_to_float_array(selected[col])
        values = values[~np.isnan(values)]
        distinct = np.unique(values)
        if distinct.size < 2:
            degenerate.append(col)
            continue

        ranks = _average_ranks(values)
        n = len(values)
        z = norm.ppf((ranks - 0.5) / n)

        order = np.argsort(values, kind="stable")
        x_sorted = values[order]
        z_sorted = z[order]
        # Collapse ties: interpolation needs a strictly increasing x, and tied
        # values already share one z (averaged rank), so keep one copy of each.
        unique_x, first_idx = np.unique(x_sorted, return_index=True)
        fits[col] = OrderNormFit(x=unique_x, z=z_sorted[first_idx])

    if degenerate:
        warnings.warn(
            f"{len(degenerate)} column(s) had fewer than 2 distinct values "
            f"and were not normalized: {degenerate!r}",
            stacklevel=2,
        )

    return fits


def _interp_with_extrapolation(
    values: "np.ndarray[Any, Any]", fit: OrderNormFit
) -> "np.ndarray[Any, Any]":
    """Linearly interpolate ``values`` against ``fit``, extrapolating past the ends.

    ``numpy.interp`` clamps values outside ``fit.x`` to the boundary ``z``;
    this instead extends the boundary segment's slope, matching
    ``predict.orderNorm()``'s behavior on out-of-range ``newdata``.
    """
    z = np.interp(values, fit.x, fit.z)
    if fit.x.size >= 2:
        lo_slope = (fit.z[1] - fit.z[0]) / (fit.x[1] - fit.x[0])
        hi_slope = (fit.z[-1] - fit.z[-2]) / (fit.x[-1] - fit.x[-2])
        below = values < fit.x[0]
        above = values > fit.x[-1]
        z = np.where(below, fit.z[0] + (values - fit.x[0]) * lo_slope, z)
        z = np.where(above, fit.z[-1] + (values - fit.x[-1]) * hi_slope, z)
    return cast("np.ndarray[Any, Any]", z)


def apply_order_norm(selected: Frame, fits: dict[str, OrderNormFit]) -> Frame:
    """Map ``selected`` onto normal quantiles using stored lookup tables.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to normalize.
    fits:
        The ``{column: OrderNormFit}`` mapping returned by `fit_order_norm()`.
        Any column of ``selected`` not present as a key (e.g. a degenerate
        column dropped at fit time) is passed through untouched.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``fits`` replaced
    in place by its normal-quantile score. Missing input values pass through
    as missing. Column names, order, and structure are unchanged.
    """
    if not fits:
        return selected

    new_columns: dict[str, "np.ndarray[Any, Any]"] = {}
    for col, fit in fits.items():
        values = _frame.series_to_float_array(selected[col])
        mask = ~np.isnan(values)
        out = np.full(values.shape, np.nan)
        out[mask] = _interp_with_extrapolation(values[mask], fit)
        new_columns[col] = out

    exprs = [
        nw.new_series(name=col, values=arr, backend=selected.implementation).alias(col)
        for col, arr in new_columns.items()
    ]
    return selected.with_columns(*exprs)


class StepOrderNorm(Step):
    """Normalize numeric columns onto a standard normal distribution by rank.

    Each selected numeric column's training values are ranked (ties averaged)
    and mapped to normal quantiles, producing a lookup table between the
    original values and their normal-quantile scores. Every selected column is
    replaced in place by its score under that table; new values falling
    outside the training range are extrapolated using the slope of the nearest
    boundary segment rather than clamped.

    Unlike `StepNormalize`, which only recenters and rescales, this reshapes
    the column's whole distribution, so it also corrects skew.

    A column with fewer than 2 distinct training values cannot be mapped onto
    a spread of normal quantiles; it is dropped from the normalization (with a
    warning) and passes through ``transform`` untouched. Missing values pass
    through as missing.

    Not a base R ``recipes`` step; this is a port of ``orderNorm()`` from the R
    ``bestNormalize`` package.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to normalize. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    id
        Stable step identifier. Defaults to ``"order_norm"``; a second step
        of this type in the same recipe becomes ``"order_norm_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its normal-quantile score:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepOrderNorm
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 4.0, 8.0, 16.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepOrderNorm(s.numeric_features()))
    recipe
    ```

    `fit()` learns each column's rank-to-normal-quantile lookup table:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` applies it in place, turning the right-skewed input into a
    symmetric, standard-normal-shaped output:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        self.columns = columns
        super().__init__(id, "order_norm")
        # Populated by ``fit()``: {column: OrderNormFit} over normalizable columns.
        self._estimates: dict[str, OrderNormFit] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn each column's rank-to-normal-quantile lookup table."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_order_norm(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Map the stored lookup tables onto ``new_data``. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepOrderNorm.transform() called before fit()")
        return apply_order_norm(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepOrderNorm.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, normalized ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepOrderNorm.schema_out() called before fit()")
        return schema_out_for_touched(self._schema_in, set(self._estimates))

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Ordered-quantile normalization is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    def required_pkgs(self) -> list[str]:
        """Ranks and normal quantiles are computed with NumPy and SciPy."""
        return ["numpy", "scipy"]

    @property
    def produces_sparse(self) -> bool:
        """The rank map moves zeros off zero; output stays dense."""
        return False
