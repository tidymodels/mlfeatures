"""Natural cubic spline basis expansion: fit + apply.

Port of R ``recipes``'s ``step_spline_natural()``, which wraps
``splines2::naturalSpline()``. The *fit* half (`fit_spline_natural()`)
inspects training data to place interior knots (by quantile, or the caller's
own explicit `knots`) and stores them alongside the training boundary
(``min``/``max``); the *apply* half (`apply_spline_natural()`) reuses those
stored knots to expand each selected column — the basis is never recomputed
from new data, so train/serve parity holds the way it does for
`mlfeatures.step_polynomial`. `StepSplineNatural` wraps both into the `Step`
lifecycle so a `Recipe` can fit and apply them.

The basis math itself (`natural_null_space()`) is a direct port of
``splines2``'s C++ core (``splines2Armadillo/NaturalSpline.h``): build the
standard cubic B-spline design matrix over the padded knot vector, then
project it through a closed-form "null space" matrix -- the basis for the
space of B-spline coefficient vectors whose second derivative vanishes at
both boundary knots (the natural-boundary constraint). ``splines2`` hardcodes
this matrix for 0, 1, and >=2 interior knots rather than solving a generic
null-space problem at runtime (see Wang & Yan, "Shape-Restricted Regression
Splines with R Package splines2", JDS 2021); this port keeps the same
case split and was verified bit-for-bit against live R `splines2` output
across all three cases, both `complete_set` settings, and out-of-boundary
extrapolation.
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
    "SplineNaturalFit",
    "fit_spline_natural",
    "apply_spline_natural",
    "StepSplineNatural",
]


@dataclass(frozen=True)
class SplineNaturalFit:
    """Per-column natural-spline basis state learned at fit time.

    Bundles everything `apply_spline_natural()` needs to reproduce the same
    basis on new data: the interior knots (placed once, from training data,
    or supplied explicitly), the training-set boundary, and the effective
    number of output columns. Marked ``frozen=True`` so the fitted state
    can't be mutated and drift train-to-serve. Unlike `mlfeatures.step_polynomial`'s
    ``PolynomialFit``, no separate transform matrix is stored: the knots
    *are* the pinned state, and the null-space matrix is a cheap, pure
    function of them, recomputed (not re-estimated) at apply time.

    Attributes
    ----------
    internal_knots
        Interior knot locations, sorted ascending. Empty for a degenerate
        column (``deg_free == 0``) or a fit with 0 interior knots.
    boundary_knots
        ``(min, max)`` of the column's non-null training values. ``(nan,
        nan)`` for a degenerate column.
    complete_set
        Whether the constant basis column is included in the output.
    deg_free
        Effective number of output columns. ``0`` for a degenerate column
        (fewer than 2 distinct training values) — those are passed through
        untouched at apply time (with a warning), the same convention as a
        degree-0 `mlfeatures.step_polynomial.PolynomialFit`.
    """

    internal_knots: "np.ndarray[Any, Any]"
    boundary_knots: tuple[float, float]
    complete_set: bool
    deg_free: int = field(default=0)


def natural_null_space(
    internal_knots: "np.ndarray[Any, Any]",
    boundary_knots: tuple[float, float],
    spline_df: int,
    standardize: bool = True,
) -> "np.ndarray[Any, Any]":
    """The closed-form null-space matrix for the natural-boundary constraint.

    Port of ``splines2Armadillo::NaturalSpline::set_null_colvecs()``. Maps the
    ``spline_df + 2``-column complete cubic B-spline basis (``spline_df``
    interior-knot-implied dimension, +2 for the intercept and linear term) down
    onto the ``spline_df``-dimensional space of coefficient vectors whose
    second derivative is zero at both boundary knots — the defining property
    of a natural spline. ``spline_df`` here is always ``len(internal_knots) +
    2`` (the *complete*-basis dimension, i.e. before dropping a column for
    ``complete_set=False``), independent of the step's ``complete_set``
    setting.
    """
    n_knots = internal_knots.size
    m = np.zeros((spline_df + 2, spline_df))
    if n_knots == 0:
        m[0, 0], m[1, 0], m[2, 0] = 3.0, 2.0, 1.0
        m[1, 1], m[2, 1], m[3, 1] = 1.0, 2.0, 3.0
    elif n_knots == 1:
        w1 = internal_knots[0] - boundary_knots[0]
        w2 = boundary_knots[1] - internal_knots[0]
        w3 = w1 + w2
        m[0, 0], m[1, 0] = 1.0 + w1 / w3, 1.0
        m[1, 1] = 1.0 / (1.0 + w3 / w1)
        m[2, 1] = 1.0
        m[3, 1] = 1.0 / (1.0 + w3 / w2)
        m[3, 2], m[4, 2] = 1.0, 1.0 + w2 / w3
    else:
        for i in range(3):
            m[i, 0] = 1.0
            m[spline_df - i + 1, spline_df - 1] = 1.0
        m[1, 1] = 1.0
        m[2, 1] = 1.0 + (internal_knots[1] - boundary_knots[0]) / (
            internal_knots[0] - boundary_knots[0]
        )
        m[spline_df - 1, spline_df - 2] = 1.0 + (
            boundary_knots[1] - internal_knots[n_knots - 2]
        ) / (boundary_knots[1] - internal_knots[n_knots - 1])
        m[spline_df, spline_df - 2] = 1.0
        if spline_df > 4:
            for j in range(spline_df - 4):
                m[j + 3, j + 2] = 1.0
    if standardize:
        m = m / m.sum(axis=0, keepdims=True)
    return m


def fit_spline_natural(
    selected: Frame,
    deg_free: int,
    complete_set: bool,
    knots: Sequence[float] | None = None,
) -> dict[str, SplineNaturalFit]:
    """Learn the natural-spline knot placement for each column.

    For each column, places ``deg_free - (2 if complete_set else 1)`` interior
    knots at quantiles of the non-null training values (or uses the caller's
    explicit `knots`, validated against that column's boundary), then stores
    them alongside the training boundary in a `SplineNaturalFit`. A column
    with fewer than 2 distinct non-null values cannot support a boundary at
    all and is **omitted from expansion** (``deg_free=0``, a warning names it),
    the same convention `mlfeatures.step_polynomial.fit_polynomial` uses for a
    constant column.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the columns to be expanded.
    deg_free
        Number of output basis columns per (non-degenerate) selected column.
        Must be an ``int`` >= 2. Ignored when `knots` is given (the interior
        knot count then follows from ``len(knots)`` instead).
    complete_set
        Whether to include the constant basis column in the output.
    knots
        Explicit interior knot locations, shared across every selected
        column, or ``None`` (default) to place them at quantiles of each
        column's own training values. Each column's knots must lie strictly
        inside that column's own training boundary — a `ValueError` names the
        offending column if not.

    Returns
    -------
    dict[str, SplineNaturalFit]
        Mapping from column name to its fitted knot state, in input column
        order. Pair with `apply_spline_natural()` to produce the expanded
        frame.

    Raises
    ------
    ValueError
        If `deg_free` is not an integer >= 2 (only checked when `knots` is
        not given), or if `knots` is invalid for some column (see
        `mlfeatures._spline.check_explicit_knots`).
    """
    _spline.check_complete_set(complete_set)
    if knots is None:
        _spline.check_deg_free(deg_free, minimum=2)

    fit_result: dict[str, SplineNaturalFit] = {}
    degenerate: list[str] = []

    for col_name in selected.columns:
        values = _frame.series_to_float_array(selected[col_name])
        values = values[~np.isnan(values)]
        if np.unique(values).size < 2:
            degenerate.append(col_name)
            fit_result[col_name] = SplineNaturalFit(
                internal_knots=np.array([]),
                boundary_knots=(float("nan"), float("nan")),
                complete_set=complete_set,
                deg_free=0,
            )
            continue

        lo, hi = _spline.boundary_knots(values)
        if knots is not None:
            interior = _spline.check_explicit_knots(knots, (lo, hi), column=col_name)
        else:
            n_interior = max(0, deg_free - (2 if complete_set else 1))
            interior = _spline.capped_interior_knots(values, n_interior)

        effective_deg_free = interior.size + (2 if complete_set else 1)
        fit_result[col_name] = SplineNaturalFit(
            internal_knots=interior,
            boundary_knots=(lo, hi),
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


def apply_spline_natural(
    data: Frame,
    fit_result: dict[str, SplineNaturalFit],
    keep_original_cols: bool = False,
) -> Frame:
    """Replace columns in ``data`` with their fitted natural-spline expansions.

    For each column in ``data``, in input order:

    - If the column is **not** in ``fit_result``, pass it through unchanged.
    - If it is in ``fit_result`` with ``deg_free > 0``, replace it in place
      (same positional slot) with ``deg_free`` basis columns named
      ``{column}_1``, ``{column}_2``, ..., ``{column}_{deg_free}``, using the
      fit-time knots — the basis is **not recomputed** from ``data``. With
      ``keep_original_cols`` the source column stays in its slot and the
      expansion follows it.
    - If its ``deg_free`` is ``0``, keep the original column and emit a
      `UserWarning` naming the column.

    New data falling outside the fit-time boundary knots is extrapolated
    **linearly** (using the raw B-spline basis value and first derivative at
    the boundary, then projected through the same null-space matrix) — this
    is a defining property of a natural spline, not a simplification, and was
    verified against R `splines2`'s own out-of-boundary output.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_spline_natural()`.
    keep_original_cols
        When ``True``, each expanded source column is left in place alongside
        its basis columns instead of being dropped.

    Returns
    -------
    Frame
        New frame with the selected columns replaced by their natural-spline
        expansions. Backend (pandas/polars) is preserved end-to-end.

    Warns
    -----
    UserWarning
        When a column's ``deg_free`` is ``0`` (fewer than 2 distinct training
        values), it is left untouched and a warning is emitted.
    """
    final_order: list[str] = []
    expand_specs: list[tuple[str, SplineNaturalFit]] = []
    to_drop: list[str] = []
    for col_name in data.columns:
        fit = fit_result.get(col_name)
        if fit is None:
            final_order.append(col_name)
            continue
        if fit.deg_free <= 0:
            warnings.warn(
                f"column {col_name!r} has fewer than 2 unique values; "
                "no natural spline expansion was applied",
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
        z = _natural_basis(data[col_name], fit)
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


def _natural_basis(
    series: nw.Series[Any], fit: SplineNaturalFit
) -> "np.ndarray[Any, Any]":
    """Evaluate a stored natural-spline basis on new data.

    Splits ``series`` into rows inside the fit-time boundary (evaluated
    through the standard B-spline design matrix) and rows outside it
    (linearly extrapolated from the exact basis value + derivative at the
    boundary knot they overshot), then projects both through the same
    `natural_null_space` matrix. Missing entries become rows of ``NaN``.
    """
    x = _frame.series_to_float_array(series)
    mask = ~np.isnan(x)
    xv = x[mask]
    lo, hi = fit.boundary_knots
    spline_df_complete = fit.internal_knots.size + 2
    null_space = natural_null_space(
        fit.internal_knots, fit.boundary_knots, spline_df_complete
    )

    complete = np.empty((xv.shape[0], spline_df_complete))
    inside = (xv >= lo) & (xv <= hi)
    if np.any(inside):
        design = _spline.bspline_design_matrix(
            xv[inside], fit.internal_knots, fit.boundary_knots, degree=3
        )
        complete[inside] = design @ null_space

    for side_mask, bnd in ((xv < lo, lo), (xv > hi, hi)):
        if np.any(side_mask):
            bnd_arr = np.array([bnd])
            b0 = _spline.bspline_design_matrix(
                bnd_arr, fit.internal_knots, fit.boundary_knots, degree=3
            )[0]
            d0 = _spline.bspline_design_matrix(
                bnd_arr, fit.internal_knots, fit.boundary_knots, degree=3, deriv=1
            )[0]
            offsets = (xv[side_mask] - bnd)[:, None]
            complete[side_mask] = (b0[None, :] + d0[None, :] * offsets) @ null_space

    natural = complete if fit.complete_set else complete[:, 1:]

    z = np.full((x.shape[0], fit.deg_free), np.nan)
    z[mask] = natural
    return z


class StepSplineNatural(Step):
    """Replace numeric columns with a natural cubic spline basis expansion.

    Each selected numeric column ``x`` is replaced by ``deg_free`` natural
    cubic spline basis columns ``{x}_1``, ``{x}_2``, ..., ``{x}_{deg_free}``,
    or kept alongside them when ``keep_original_cols`` is ``True``. A natural
    spline is a piecewise cubic polynomial constrained to be linear beyond its
    boundary knots, which avoids the wild swings an ordinary polynomial or
    unconstrained cubic spline can show near the edges of the data.

    Interior knots are placed at quantiles of each column's own training
    values (or at the caller-supplied `knots`, shared across every selected
    column). A column with fewer than 2 distinct training values cannot
    support a boundary at all; `transform()` leaves such a column untouched
    and emits a `UserWarning`. New data outside the fit-time boundary is
    extrapolated linearly, not as a wild polynomial continuation — this is
    what "natural" means for this basis.

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
        Must be an integer >= 2. Default ``10``. Ignored when `knots` is
        given.
    complete_set
        Whether to include the constant basis column in the output. Default
        ``False``.
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
        Stable step identifier. Defaults to ``"spline_natural"``; a second
        step of this type in the same recipe becomes ``"spline_natural_2"``,
        and passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its natural-spline basis
    columns ``{x}_1 ... {x}_deg_free``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepSplineNatural
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepSplineNatural(s.numeric_features(), deg_free=3))
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
    recipe = Recipe().add(StepSplineNatural(s.numeric_features(), deg_free=4))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside its basis columns:

    ```{python}
    recipe = Recipe().add(
        StepSplineNatural(s.numeric_features(), deg_free=3, keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    #: Populated by ``fit()``: per-column fitted knots and basis settings.
    _estimates: dict[str, SplineNaturalFit] | None

    def __init__(
        self,
        columns: Selector,
        *,
        deg_free: int = 10,
        complete_set: bool = False,
        knots: Sequence[float] | None = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.deg_free = deg_free
        self.complete_set = complete_set
        self.knots = knots
        super().__init__(
            id,
            "spline_natural",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the natural-spline knot placement for each selected column."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_spline_natural(
            selected, self.deg_free, self.complete_set, self.knots
        )
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        self._schema_out, self._output_parents = expand_columns_schema(
            self._schema_in, self._estimates, lambda fit: fit.deg_free
        )
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Replace each selected column with its natural-spline expansion."""
        if self._estimates is None:
            raise RuntimeError("StepSplineNatural.transform() called before fit()")
        return apply_spline_natural(new_data, self._estimates, self.keep_original_cols)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepSplineNatural.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the basis columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepSplineNatural.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each basis column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepSplineNatural.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Natural spline expansion is only meaningful on numeric columns."""
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
        """The B-spline design matrix is built with SciPy."""
        return ["scipy"]

    @property
    def produces_sparse(self) -> bool:
        """A natural spline basis is dense by construction."""
        return False
