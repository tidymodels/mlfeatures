"""Monotone (I-spline) spline basis expansion: fit + apply.

Port of R ``recipes``'s ``step_spline_monotone()``, which wraps
``splines2::iSpline()``. The *fit* half (`fit_spline_monotone()`) inspects
training data to place interior knots (by quantile, or the caller's own
explicit `knots`) and stores them alongside the training boundary; the
*apply* half (`apply_spline_monotone()`) reuses those stored knots to expand
each selected column: the basis is never recomputed from new data, matching
`mlfeatures.step_polynomial` and `mlfeatures.step_spline_natural`. `StepSplineMonotone`
wraps both into the `Step` lifecycle so a `Recipe` can fit and apply them.

Unlike scikit-learn's ``SplineTransformer`` (B-spline only, no shape
constraints), each basis column here is non-negative and non-decreasing:
when used with a fitting procedure that produces non-negative coefficients,
the resulting fit is guaranteed monotonic in the column. This is a genuine
gap in the Python ecosystem, not a wrapper around an existing transformer.

The basis math falls out of an exact identity rather than needing its own
closed form the way `mlfeatures.step_spline_natural` does: an I-spline basis
function is by definition the integral of an M-spline basis function, and an
M-spline is just a rescaled B-spline (``M_i(x) = order / (t[i+order] - t[i])
* B_i(x)``, the standard de Boor normalization). SciPy's
``BSpline.antiderivative()`` computes the *exact* antiderivative of a
B-spline (via the standard coefficient cumulative-sum identity, not numerical
quadrature), so the I-spline basis is a direct, exact port with no bespoke
matrix to derive, verified bit-for-bit against live R `splines2` output,
including out-of-boundary extrapolation, which needs no special-casing here
(unlike the natural spline): SciPy's default polynomial-continuation
extrapolation reproduces R's own (documented as potentially ill-conditioned)
behavior exactly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw
import numpy as np

from mlfeatures import _frame, _spline
from mlfeatures._basis_expansion import (
    expand_columns_schema,
    update_roles_for_expansion,
)
from mlfeatures._dependencies import import_optional
from mlfeatures.step import Step
from mlfeatures.types import (
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

__all__ = [
    "SplineMonotoneFit",
    "fit_spline_monotone",
    "apply_spline_monotone",
    "StepSplineMonotone",
]


@dataclass(frozen=True)
class SplineMonotoneFit:
    """Per-column monotone-spline (I-spline) basis state learned at fit time.

    Bundles everything `apply_spline_monotone()` needs to reproduce the same
    basis on new data: the interior knots, the training-set boundary, the
    underlying M-spline degree, and the effective number of output columns.
    Marked ``frozen=True`` so the fitted state can't be mutated and drift
    train-to-serve. As with `mlfeatures.step_spline_natural`'s ``SplineNaturalFit``,
    no separate transform matrix is stored: the knots are the pinned state.

    Attributes
    ----------
    internal_knots
        Interior knot locations, sorted ascending. Empty for a degenerate
        column (``deg_free == 0``) or a fit with 0 interior knots.
    boundary_knots
        ``(min, max)`` of the column's non-null training values. ``(nan,
        nan)`` for a degenerate column.
    degree
        Degree of the underlying M-spline (the I-spline's pieces have degree
        ``degree + 1``).
    complete_set
        Whether the first basis column is included in the output.
    deg_free
        Effective number of output columns. ``0`` for a degenerate column
        (fewer than 2 distinct training values): those are passed through
        untouched at apply time (with a warning).
    """

    internal_knots: "np.ndarray[Any, Any]"
    boundary_knots: tuple[float, float]
    degree: int
    complete_set: bool
    deg_free: int = field(default=0)


def fit_spline_monotone(
    selected: Frame,
    deg_free: int,
    degree: int,
    complete_set: bool,
    knots: Sequence[float] | None = None,
) -> dict[str, SplineMonotoneFit]:
    """Learn the monotone-spline (I-spline) knot placement for each column.

    For each column, places
    ``deg_free - degree - (1 if complete_set else 0)`` interior knots at
    quantiles of the non-null training values (or uses the caller's explicit
    `knots`, validated against that column's boundary), then stores them
    alongside the training boundary in a `SplineMonotoneFit`. A column with
    fewer than 2 distinct non-null values cannot support a boundary at all
    and is **omitted from expansion** (``deg_free=0``, a warning names it),
    the same convention `mlfeatures.step_polynomial.fit_polynomial` uses for a
    constant column.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the columns to be expanded.
    deg_free
        Number of output basis columns per (non-degenerate) selected column.
        Must be an ``int`` at least ``degree + (1 if complete_set else 0)``
        (0 interior knots). Ignored when `knots` is given.
    degree
        Degree of the underlying M-spline. Must be a non-negative ``int``.
    complete_set
        Whether to include the first basis column in the output.
    knots
        Explicit interior knot locations, shared across every selected
        column, or ``None`` (default) to place them at quantiles of each
        column's own training values.

    Returns
    -------
    dict[str, SplineMonotoneFit]
        Mapping from column name to its fitted knot state, in input column
        order. Pair with `apply_spline_monotone()` to produce the expanded
        frame.

    Raises
    ------
    ValueError
        If `degree` is not a non-negative integer, if `deg_free` is too small
        to admit 0 interior knots (only checked when `knots` is not given),
        or if `knots` is invalid for some column (see
        `mlfeatures._spline.check_explicit_knots`).
    """
    _spline.check_degree(degree)
    _spline.check_complete_set(complete_set)
    if knots is None:
        minimum = degree + (1 if complete_set else 0)
        _spline.check_deg_free(deg_free, minimum=minimum)

    fit_result: dict[str, SplineMonotoneFit] = {}
    degenerate: list[str] = []

    for col_name in selected.columns:
        values = _frame.series_to_float_array(selected[col_name])
        values = values[~np.isnan(values)]
        if np.unique(values).size < 2:
            degenerate.append(col_name)
            fit_result[col_name] = SplineMonotoneFit(
                internal_knots=np.array([]),
                boundary_knots=(float("nan"), float("nan")),
                degree=degree,
                complete_set=complete_set,
                deg_free=0,
            )
            continue

        lo, hi = _spline.boundary_knots(values)
        if knots is not None:
            interior = _spline.check_explicit_knots(knots, (lo, hi), column=col_name)
        else:
            n_interior = max(0, deg_free - degree - (1 if complete_set else 0))
            interior = _spline.capped_interior_knots(values, n_interior)

        effective_deg_free = interior.size + degree + (1 if complete_set else 0)
        fit_result[col_name] = SplineMonotoneFit(
            internal_knots=interior,
            boundary_knots=(lo, hi),
            degree=degree,
            complete_set=complete_set,
            deg_free=effective_deg_free,
        )

    if degenerate:
        warnings.warn(
            f"{len(degenerate)} column(s) had fewer than 2 distinct values "
            f"and were not expanded: {degenerate!r}",
            UserWarning,
            stacklevel=2,
        )

    return fit_result


def apply_spline_monotone(
    data: Frame,
    fit_result: dict[str, SplineMonotoneFit],
    keep_original_cols: bool = False,
) -> Frame:
    """Replace columns in ``data`` with their fitted monotone-spline expansions.

    For each column in ``data``, in input order:

    - If the column is **not** in ``fit_result``, pass it through unchanged.
    - If it is in ``fit_result`` with ``deg_free > 0``, replace it in place
      (same positional slot) with ``deg_free`` basis columns named
      ``{column}_1``, ``{column}_2``, ..., ``{column}_{deg_free}``, using the
      fit-time knots: the basis is **not recomputed** from ``data``. With
      ``keep_original_cols`` the source column stays in its slot and the
      expansion follows it.
    - If its ``deg_free`` is ``0``, keep the original column and emit a
      `UserWarning` naming the column.

    New data falling outside the fit-time boundary knots continues the local
    polynomial piece rather than clamping or linearly extrapolating (matching
    R `splines2`'s own documented behavior): the basis can become
    non-monotonic and leave ``[0, 1]`` out there, since the shape guarantee
    is only made within the training range.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_spline_monotone()`.
    keep_original_cols
        When ``True``, each expanded source column is left in place alongside
        its basis columns instead of being dropped.

    Returns
    -------
    Frame
        New frame with the selected columns replaced by their monotone-spline
        expansions. Backend (pandas/polars) is preserved end-to-end.

    Warns
    -----
    UserWarning
        When a column's ``deg_free`` is ``0`` (fewer than 2 distinct training
        values), it is left untouched and a warning is emitted.
    """
    final_order: list[str] = []
    expand_specs: list[tuple[str, SplineMonotoneFit]] = []
    to_drop: list[str] = []
    for col_name in data.columns:
        fit = fit_result.get(col_name)
        if fit is None:
            final_order.append(col_name)
            continue
        if fit.deg_free <= 0:
            warnings.warn(
                f"column {col_name!r} has fewer than 2 unique values; "
                "no monotone spline expansion was applied",
                UserWarning,
                stacklevel=2,
            )
            final_order.append(col_name)
            continue
        expand_specs.append((col_name, fit))
        if keep_original_cols:
            final_order.append(col_name)
        else:
            to_drop.append(col_name)
        final_order.extend(f"{col_name}_{i}" for i in range(1, fit.deg_free + 1))

    new_series: list[nw.Series[Any]] = []
    for col_name, fit in expand_specs:
        z = _ispline_basis(data[col_name], fit)
        for i in range(fit.deg_free):
            new_series.append(
                nw.new_series(
                    name=f"{col_name}_{i + 1}",
                    values=z[:, i],
                    backend=data.implementation,
                )
            )

    if new_series:
        data = data.with_columns(*new_series)
    if to_drop:
        data = data.drop(to_drop)
    return data.select(final_order)


# --- Private helper -------------------------------------------------------


def _ispline_basis(
    series: nw.Series[Any], fit: SplineMonotoneFit
) -> "np.ndarray[Any, Any]":
    """Evaluate a stored I-spline basis on new data.

    Each I-spline basis column is the exact antiderivative of a rescaled
    B-spline (M-spline) basis function, shifted so it is zero at the lower
    boundary knot. ``extrapolate=True`` on the underlying `BSpline` is what
    reproduces R `splines2`'s raw polynomial-continuation behavior past the
    boundary (verified bit-for-bit against live R output): no special-casing
    needed, unlike the natural spline. Missing entries become rows of ``NaN``.
    """
    bspline_cls = import_optional(
        "scipy.interpolate", purpose="for spline basis construction"
    ).BSpline

    x = _frame.series_to_float_array(series)
    mask = ~np.isnan(x)
    xv = x[mask]

    order = fit.degree + 1
    t = _spline.padded_knot_vector(fit.internal_knots, fit.boundary_knots, fit.degree)
    n_bases_complete = len(t) - order
    lo = fit.boundary_knots[0]

    complete = np.empty((xv.shape[0], n_bases_complete))
    for i in range(n_bases_complete):
        scale = order / (t[i + order] - t[i])
        coef = np.zeros(n_bases_complete)
        coef[i] = scale
        m_spline = bspline_cls(t, coef, fit.degree, extrapolate=True)
        antiderivative = m_spline.antiderivative()
        complete[:, i] = antiderivative(xv) - antiderivative(lo)

    monotone = complete if fit.complete_set else complete[:, 1:]

    z = np.full((x.shape[0], fit.deg_free), np.nan)
    z[mask] = monotone
    return z


class StepSplineMonotone(Step):
    """Replace numeric columns with a monotone (I-spline) basis expansion.

    Each selected numeric column ``x`` is replaced by ``deg_free`` I-spline
    basis columns ``{x}_1``, ``{x}_2``, ..., ``{x}_{deg_free}``, or kept
    alongside them when ``keep_original_cols`` is ``True``. Every basis column
    is non-negative and non-decreasing in ``x``; fit with a procedure that
    keeps its coefficients non-negative (e.g. non-negative least squares), the
    resulting model is monotonic in ``x``.

    Interior knots are placed at quantiles of each column's own training
    values (or at the caller-supplied `knots`, shared across every selected
    column). A column with fewer than 2 distinct training values cannot
    support a boundary at all; `transform()` leaves such a column untouched
    and emits a `UserWarning`. New data outside the fit-time boundary
    continues the local polynomial piece and can leave ``[0, 1]`` or lose
    monotonicity: the shape guarantee only holds within the training range.

    See [Splines](https://feaz-book.com/numeric-splines) in *Feature Engineering
    A to Z* for background on this technique.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to expand. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    deg_free
        Number of output basis columns per (non-degenerate) selected column.
        Must be an integer at least ``degree`` (plus 1 if `complete_set`).
        Default ``10``. Ignored when `knots` is given.
    degree
        Degree of the underlying M-spline; the I-spline's pieces have degree
        ``degree + 1``. Must be a non-negative integer. Default ``3``.
    complete_set
        Whether to include the first basis column in the output. Default
        ``True``.
    knots
        Explicit interior knot locations, shared across every selected
        column, instead of the default quantile placement. Each column's
        knots must lie strictly inside that column's own training boundary.
        Default ``None`` (quantile placement).
    keep_original_cols
        If ``True``, the source columns are kept alongside their basis
        columns instead of being dropped. Default ``False``.
    role
        Role assigned to the new basis columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"spline_monotone"``; a second
        step of this type in the same recipe becomes ``"spline_monotone_2"``,
        and passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its monotone-spline basis
    columns ``{x}_1 ... {x}_deg_free``, each rising from 0 to 1:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepSplineMonotone
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepSplineMonotone(s.numeric_features(), deg_free=4))
    recipe
    ```

    `fit()` places the interior knots at quantiles of the training data:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the column with its basis columns:

    ```{python}
    fitted.transform(df)
    ```

    A higher ``deg_free`` produces more, more flexible basis columns:

    ```{python}
    recipe = Recipe().add(StepSplineMonotone(s.numeric_features(), deg_free=6))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside its basis columns:

    ```{python}
    recipe = Recipe().add(
        StepSplineMonotone(s.numeric_features(), deg_free=4, keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    #: Populated by ``fit()``: per-column fitted knots and basis settings.
    _estimates: dict[str, SplineMonotoneFit] | None

    def __init__(
        self,
        columns: Selector,
        *,
        deg_free: int = 10,
        degree: int = 3,
        complete_set: bool = True,
        knots: Sequence[float] | None = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.deg_free = deg_free
        self.degree = degree
        self.complete_set = complete_set
        self.knots = knots
        super().__init__(
            id,
            "spline_monotone",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the monotone-spline knot placement for each selected column."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_spline_monotone(
            selected, self.deg_free, self.degree, self.complete_set, self.knots
        )
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        self._schema_out, self._output_parents = expand_columns_schema(
            self._schema_in, self._estimates, lambda fit: fit.deg_free
        )
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Replace each selected column with its monotone-spline expansion."""
        if self._estimates is None:
            raise RuntimeError("StepSplineMonotone.transform() called before fit()")
        return apply_spline_monotone(new_data, self._estimates, self.keep_original_cols)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepSplineMonotone.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the basis columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepSplineMonotone.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each basis column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError(
                "StepSplineMonotone.output_parents() called before fit()"
            )
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Monotone spline expansion is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._output_parents is None:
            return
        update_roles_for_expansion(
            roles,
            self._output_parents,
            output_role=self.output_role,
            keep_original_cols=self.keep_original_cols,
        )

    def required_pkgs(self) -> list[str]:
        """The I-spline basis is built with SciPy."""
        return ["scipy"]

    @property
    def produces_sparse(self) -> bool:
        """A monotone spline basis is dense by construction."""
        return False
