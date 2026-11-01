"""Orthogonal polynomial expansion: fit + apply.

The *fit* half (`fit_polynomial()`) inspects training data to pick a
polynomial degree per column and **learns the orthogonal basis** (centering
mean + QR transform matrix) for each fittable column. The *apply* half
(`apply_polynomial()`) reuses those statistics to expand each selected
column; the basis is never recomputed from new data: train/serve parity with
R's `stats::poly()`. `StepPolynomial` wraps both into the
`Step` lifecycle so a `Recipe` can fit
and apply them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np
from numpy.polynomial.polynomial import polyvander

from mlfeatures import _frame
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
    "PolynomialFit",
    "fit_polynomial",
    "apply_polynomial",
    "StepPolynomial",
]


@dataclass(frozen=True)
class PolynomialFit:
    """Per-column orthogonal polynomial basis learned at fit time.

    Bundles everything `apply_polynomial()` needs to reproduce R's
    `stats::poly()` output on new data: the effective polynomial degree,
    the training-set mean used for centering, and the linear transform
    derived from the QR factorization of the training-data Vandermonde so
    the basis is **pinned to the training set** (not recomputed at apply
    time). Marked ``frozen=True`` so the fitted state can't be mutated and
    drift train-to-serve.

    Attributes
    ----------
    degree
        Effective polynomial degree for this column. ``0`` for constant /
        all-null columns where no orthogonal expansion is possible: those
        are passed through untouched at apply time (with a warning).
    mean
        Training-set mean of the column's non-null values. Applied as
        ``x_centered = x - mean`` before evaluating the basis on new data.
    transform
        Matrix of shape ``(degree + 1, degree + 1)`` such that the
        orthogonal polynomial values for any new ``x`` are
        ``polyvander(x - mean, degree) @ transform`` (the leading constant
        column is then dropped). Built once at fit as
        ``inv(R) @ diag(sign(diag(R)))`` from the QR of the training-data
        Vandermonde. Empty for ``degree == 0``.
    """

    degree: int
    mean: float
    transform: np.ndarray[Any, Any] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )


def fit_polynomial(selected: Frame, degree: int) -> dict[str, PolynomialFit]:
    """Learn the orthogonal polynomial basis for each column.

    For each column, picks an *effective* degree of
    ``min(degree, n_unique_non_null - 1)``, clamped at zero: a column with
    ``k`` unique values can support at most ``k - 1`` orthogonal polynomial
    terms before the Vandermonde becomes rank-deficient. For columns whose
    effective degree is positive, computes the centering mean and the QR
    transform matrix on the non-null training values and stores them in a
    `PolynomialFit`; constant / all-null columns get an empty fit
    that `apply_polynomial()` passes through (with a warning).

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the columns to
        be expanded. Native frames are wrapped via `mlfeatures._frame.from_native()`
        at the recipe boundary.
    degree
        Maximum polynomial degree. Must be an ``int`` strictly greater than
        zero. Booleans are rejected even though ``bool`` is a subclass of
        ``int`` in Python: ``degree=True`` is almost always a caller bug.

    Returns
    -------
    dict[str, PolynomialFit]
        Mapping from column name to its fitted polynomial state, in input
        column order. Pair with `apply_polynomial()` to produce the
        expanded frame.

    Raises
    ------
    ValueError
        If ``degree`` is not a positive integer.

    The effective degree is capped at ``n_unique_non_null - 1``, so a column
    with 3 unique values supports at most degree 2 even if a larger ``degree``
    is requested. See `StepPolynomial` for a runnable end-to-end example.
    """
    # `bool` is technically an `int` subclass in Python (True == 1, False == 0),
    # so a plain `isinstance(degree, int)` check would let `degree=True` through.
    # Reject it explicitly. The `pyright: ignore` is because pyright trusts
    # the `int` annotation, but the runtime check is still worth keeping for
    # users who pass an unchecked value (e.g. from CLI args, YAML configs).
    if (
        isinstance(degree, bool)
        or not isinstance(degree, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or degree <= 0
    ):
        raise ValueError(
            f"'degree' must be an integer greater than zero, got {degree!r}"
        )

    fit_result: dict[str, PolynomialFit] = {}
    for col_name in selected.columns:
        # Drop nulls first: a null value is not a fittable level, and
        # narwhals' `n_unique` counts null as its own value otherwise.
        col = selected[col_name]
        n_unique = col.drop_nulls().n_unique()
        # Cap the requested degree at what the data can actually support, and
        # never go below zero (an empty / all-null column would otherwise give
        # a negative effective degree).
        effective_degree = max(0, min(degree, n_unique - 1))
        if effective_degree == 0:
            # Nothing to fit; record a degree-0 sentinel so apply_polynomial
            # can warn and pass through.
            fit_result[col_name] = PolynomialFit(degree=0, mean=float("nan"))
            continue
        # Coerce to float ndarray for the QR. narwhals nulls become NaN here;
        # we drop them so the mean / Vandermonde / QR see complete rows only.
        x = _frame.series_to_float_array(col)
        x_complete = x[~np.isnan(x)]
        mean = float(x_complete.mean())
        xc = x_complete - mean
        # Vandermonde of the centered training values: [1, xc, xc^2, ..., xc^d].
        # Cast keeps pyright happy (polyvander's stubs are heavily overloaded);
        # mypy already infers a concrete ndarray, hence the targeted ignore.
        vander = cast(  # type: ignore[redundant-cast]
            "np.ndarray[Any, Any]", polyvander(xc, effective_degree)
        )
        # QR factorization: V = Q @ R. At apply time we want
        #     Z = polyvander(x_new - mean, d) @ inv(R) @ diag(sign(diag(R)))
        # because that equals Q_new * sign(diag(R)) (i.e. the same basis the
        # training values would have produced, but evaluated at the new x).
        # We precompute `transform = inv(R) @ diag(sign(diag(R)))` here so
        # apply_polynomial only has to do a single matmul.
        _, r = np.linalg.qr(vander)
        # The cast pins `diag`'s partially-unknown overloads for pyright. mypy
        # finds it redundant under the numpy 2.4 stubs but not under the 2.2
        # stubs (the newest available on Python 3.10), hence `unused-ignore`.
        diag_r = cast("np.ndarray[Any, Any]", np.diag(r))  # type: ignore[redundant-cast,unused-ignore]
        sign_diag = np.sign(diag_r)
        # `solve(R, I)` is numerically preferable to `inv(R)` and just as cheap
        # here since R is small (degree + 1 square).
        inv_r = np.linalg.solve(r, np.eye(r.shape[0]))
        transform = inv_r * sign_diag[None, :]  # column-scale by sign_diag
        fit_result[col_name] = PolynomialFit(
            degree=effective_degree,
            mean=mean,
            transform=transform,
        )
    return fit_result


def apply_polynomial(
    data: Frame,
    fit_result: dict[str, PolynomialFit],
    keep_original_cols: bool = False,
) -> Frame:
    """Replace columns in ``data`` with their fitted orthogonal polynomial expansions.

    For each column in ``data``, in input order:

    - If the column is **not** in ``fit_result``, pass it through unchanged.
    - If it is in ``fit_result`` with effective degree ``d > 0``, replace it
      in place (same positional slot) with ``d`` orthogonal polynomial
      columns named ``{column}_1``, ``{column}_2``, ..., ``{column}_d``,
      using the fit-time mean and transform matrix: the basis is **not
      recomputed** from ``data``. With ``keep_original_cols`` the source
      column stays in its slot and the expansion follows it.
    - If its effective degree is ``0``, keep the original column and emit a
      `UserWarning` naming the column. The caller can chain a
      zero-variance drop step to remove such columns.

    Columns named in ``fit_result`` that are absent from ``data`` are silently
    skipped: same schema-drift tolerance as
    `mlfeatures._frame.apply_drop()`.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_polynomial()`: per-column `PolynomialFit`
        objects carrying the training-set mean and basis transform matrix.
    keep_original_cols
        When ``True``, each expanded source column is left in place alongside
        its polynomial terms instead of being dropped.

    Returns
    -------
    Frame
        New frame with the selected columns replaced by their polynomial
        expansions. Backend (pandas/polars) is preserved end-to-end.

    Warns
    -----
    UserWarning
        When a column's effective degree is ``0`` (so no expansion is
        possible) the column is left untouched and a warning is emitted.
    """
    # Plan the final column ordering and which columns to expand. We add new
    # polynomial columns via `with_columns` (which preserves the frame's row
    # alignment: `nw.concat(how="horizontal")` on pandas is index-based and
    # silently produces NaN when the frame has a non-default index, e.g. after
    # a train/test split), then drop the expanded source columns, and finally
    # `.select(final_order)` to slot the expansions into their original spot.
    final_order: list[str] = []
    expand_specs: list[tuple[str, PolynomialFit]] = []
    to_drop: list[str] = []
    for col_name in data.columns:
        fit = fit_result.get(col_name)
        if fit is None:
            # Column wasn't selected for polynomial expansion: pass through.
            final_order.append(col_name)
            continue
        if fit.degree <= 0:
            # Constant or all-null column at fit time: nothing meaningful to
            # expand to. Keep the column unchanged and warn so the caller
            # knows the polynomial step had no effect on it.
            warnings.warn(
                f"column {col_name!r} has fewer than 2 unique values; "
                "no polynomial expansion was applied",
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
        final_order.extend(f"{col_name}_{i}" for i in range(1, fit.degree + 1))

    new_series: list[nw.Series[Any]] = []
    for col_name, fit in expand_specs:
        z = _apply_ortho_poly(data[col_name], fit)
        for i in range(fit.degree):
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


def _apply_ortho_poly(
    series: nw.Series[Any], fit: PolynomialFit
) -> np.ndarray[Any, Any]:
    """Evaluate a stored orthogonal polynomial basis on new data.

    Computes ``polyvander(x - fit.mean, fit.degree) @ fit.transform`` on the
    non-null rows and drops the leading constant column, returning a full-size
    ``(n, fit.degree)`` array with NaN rows wherever the input was null. The
    centering mean and transform matrix come from the training-set QR stored
    in ``fit``: they are not re-estimated from ``series``.

    Parameters
    ----------
    series
        Single narwhals `Series` of numeric values. Missing entries
        (narwhals nulls / NumPy NaNs) become rows of ``NaN`` in the output;
        every other row is unaffected by their presence.
    fit
        Stored basis from `fit_polynomial()`. ``fit.degree`` must be
        positive: the caller (``apply_polynomial``) handles the degree-0 case
        before invoking this helper.
    """
    # narwhals nulls become NumPy NaN at the to_numpy() boundary; we mask
    # them out so they don't contaminate the Vandermonde with NaN powers.
    x = _frame.series_to_float_array(series)
    mask = ~np.isnan(x)
    xc_complete = x[mask] - fit.mean
    # `polyvander` overloads confuse pyright; the cast pins it. mypy already
    # infers the concrete ndarray, hence the redundant-cast ignore.
    vander = cast(  # type: ignore[redundant-cast]
        "np.ndarray[Any, Any]", polyvander(xc_complete, fit.degree)
    )
    # The single matmul is what makes train/serve parity work: `fit.transform`
    # was built from the training-data QR, so this evaluates the **same**
    # orthogonal basis the training values produced, at the new x's.
    z_complete = vander @ fit.transform
    # Drop the constant (degree-0) column: we keep only the polynomial terms.
    z_complete = z_complete[:, 1:]
    # Re-embed into a full-size, NaN-prefilled array so output rows align with
    # the input series row-for-row.
    z = np.full((x.shape[0], fit.degree), np.nan)
    z[mask] = z_complete
    return z


class StepPolynomial(Step):
    """Replace numeric columns with their orthogonal polynomial expansion.

    Each selected numeric column ``x`` is replaced by up to ``degree``
    orthogonal polynomial columns ``{x}_1``, ``{x}_2``, ..., ``{x}_degree``, or
    kept alongside them when ``keep_original_cols`` is ``True``. The orthogonal
    basis is a centering mean plus a QR transform matrix.

    The effective degree per column is capped at ``n_unique_non_null - 1``: a
    column with ``k`` unique values can support at most ``k - 1`` orthogonal
    polynomial terms. Constant (or all-null) columns get an effective degree
    of ``0``; `transform()` leaves those columns untouched and emits a
    warning. The caller can chain a zero-variance drop step to clean them up.

    See [Polynomial Expansion](https://feaz-book.com/numeric-polynomial) in
    *Feature Engineering A to Z* for background on this technique.

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
    degree
        Maximum polynomial degree. Must be a positive integer. Default ``2``.
        Each column's effective degree is the smaller of ``degree`` and
        ``n_unique_non_null - 1``.
    keep_original_cols
        If ``True``, the source columns are kept alongside their polynomial
        terms instead of being dropped. Default ``False``.
    role
        Role assigned to the new polynomial columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"polynomial"``; a second step
        of this type in the same recipe becomes ``"polynomial_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its orthogonal polynomial
    terms ``{x}_1 ... {x}_degree``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepPolynomial
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepPolynomial(s.numeric_features(), degree=2))
    recipe
    ```

    `fit()` learns the orthogonal basis for each column:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the column with its polynomial terms:

    ```{python}
    fitted.transform(df)
    ```

    A higher ``degree`` produces more terms:

    ```{python}
    recipe = Recipe().add(StepPolynomial(s.numeric_features(), degree=3))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside its terms:

    ```{python}
    recipe = Recipe().add(
        StepPolynomial(s.numeric_features(), degree=2, keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    #: Populated by ``fit()``: per-column polynomial degree and centering mean.
    _estimates: dict[str, PolynomialFit] | None

    def __init__(
        self,
        columns: Selector,
        *,
        degree: int = 2,
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.degree = degree
        super().__init__(
            id, "polynomial", keep_original_cols=keep_original_cols, output_role=role
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the polynomial basis for each selected column on training data."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_polynomial(selected, self.degree)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        # Degree-0 columns stay put with their original dtype; the rest become
        # ``degree`` Float64 columns named {col}_1..{col}_d (same layout as
        # ``apply_polynomial`` produces). Each expansion column descends from
        # its single source column; a degree-0 column is a pass-through (kept
        # unchanged), not a newly produced column, so it is excluded from
        # output_parents.
        self._schema_out, self._output_parents = expand_columns_schema(
            self._schema_in, self._estimates, lambda fit: fit.degree
        )
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Replace each selected column with its polynomial expansion."""
        if self._estimates is None:
            raise RuntimeError("StepPolynomial.transform() called before fit()")
        return apply_polynomial(new_data, self._estimates, self.keep_original_cols)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepPolynomial.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the polynomial columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepPolynomial.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each expansion column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepPolynomial.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Orthogonal polynomial expansion is only meaningful on numeric columns."""
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

    @property
    def produces_sparse(self) -> bool:
        """Orthogonal polynomial bases are dense by construction."""
        return False
