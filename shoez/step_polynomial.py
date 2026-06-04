"""Orthogonal polynomial expansion: fit + apply.

The *fit* half (:func:`fit_polynomial`) inspects training data to pick a
polynomial degree per column, capped by how many unique values that column
actually has. The *apply* half (:func:`apply_polynomial`) replaces each
selected column with a set of orthogonal polynomial basis expansion whose math
matches R's :func:`stats::poly`. :class:`StepPolynomial` wraps both into the
:class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can fit
and apply them.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np
from numpy.polynomial.polynomial import polyvander

from shoez import _frame
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

__all__ = ["fit_polynomial", "apply_polynomial", "StepPolynomial"]


def fit_polynomial(features: Frame, degree: int) -> dict[str, int]:
    """Choose a polynomial degree for each column.

    The *effective* degree for a column is ``min(degree, n_unique_non_null - 1)``,
    clamped at zero. The cap matters because a column with ``k`` unique values
    can support at most ``k - 1`` orthogonal polynomial terms — beyond that the
    Vandermonde matrix is rank-deficient and the QR factorization produces
    duplicate or zero columns. A constant column (one unique value) or an
    all-null column lands at an effective degree of ``0``; :func:`apply_polynomial`
    leaves those columns alone and warns rather than expanding them.

    Parameters
    ----------
    features
        Internal :data:`~shoez.types.Frame` (narwhals) holding the columns to
        be expanded. Native frames are wrapped via :func:`shoez._frame.from_native`
        at the recipe boundary.
    degree
        Maximum polynomial degree. Must be an ``int`` strictly greater than
        zero. Booleans are rejected even though ``bool`` is a subclass of
        ``int`` in Python — ``degree=True`` is almost always a caller bug.

    Returns
    -------
    dict[str, int]
        Mapping from column name to effective degree, in input column order
        (dicts preserve insertion order in Python 3.7+). Pair with
        :func:`apply_polynomial` to produce the expanded frame.

    Raises
    ------
    ValueError
        If ``degree`` is not a positive integer.

    Examples
    --------
    >>> import narwhals.stable.v2 as nw
    >>> import pandas as pd
    >>> frame = nw.from_native(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    >>> fit_polynomial(frame, degree=5)
    {'x': 2}
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
            f"`degree` must be an integer greater than zero, got {degree!r}."
        )

    fit_result: dict[str, int] = {}
    for col_name in features.columns:
        # Drop nulls first: a null observation is not a fittable level, and
        # narwhals' `n_unique` counts null as its own value otherwise.
        n_unique = features[col_name].drop_nulls().n_unique()
        # Cap the requested degree at what the data can actually support, and
        # never go below zero (an empty / all-null column would otherwise give
        # a negative effective degree).
        fit_result[col_name] = max(0, min(degree, n_unique - 1))
    return fit_result


def apply_polynomial(data: Frame, fit_result: dict[str, int]) -> Frame:
    """Replace columns in ``data`` with orthogonal polynomial expansions.

    For each column in ``data``, in input order:

    - If the column is **not** in ``fit_result``, pass it through unchanged.
    - If it is in ``fit_result`` with effective degree ``d > 0``, replace it
      in place (same positional slot) with ``d`` orthogonal polynomial
      columns named ``{column}_1``, ``{column}_2``, ..., ``{column}_d``.
    - If its effective degree is ``0``, keep the original column and emit a
      :class:`UserWarning` naming the column. The caller can chain a
      zero-variance drop step to remove such columns.

    Columns named in ``fit_result`` that are absent from ``data`` are silently
    skipped — same schema-drift tolerance as
    :func:`shoez._frame.apply_drop` (see ``design/lifecycle.md``).

    Parameters
    ----------
    data
        Internal :data:`~shoez.types.Frame` to transform.
    fit_result
        Output of :func:`fit_polynomial`: ``{column_name: effective_degree}``.

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
    # Build the output as an ordered list of "pieces" (each piece is itself a
    # narwhals DataFrame with one or more columns), then concat horizontally.
    # This keeps the whole function in narwhals — no per-backend code paths
    # and no manual column-by-column reordering.
    pieces: list[Frame] = []
    for col_name in data.columns:
        d = fit_result.get(col_name)
        if d is None:
            # Column wasn't selected for polynomial expansion — pass through.
            pieces.append(data.select([col_name]))
            continue
        if d <= 0:
            # Constant or all-null column: nothing meaningful to expand to.
            # Keep the column unchanged and warn so the caller knows the
            # polynomial step had no effect on it.
            warnings.warn(
                f"Column {col_name!r} has fewer than two unique values; "
                "no polynomial expansion was applied.",
                UserWarning,
                stacklevel=2,
            )
            pieces.append(data.select([col_name]))
            continue
        pieces.append(_ortho_poly(data[col_name], d))
    # Narwhals' stable.v2 alias chain confuses mypy when free functions like
    # `concat` return a generic `FrameT`; the runtime type is the stable.v2
    # DataFrame (covered by tests), so the ignore is safe.
    return nw.concat(pieces, how="horizontal")  # type: ignore[return-value]


# --- Private helper -------------------------------------------------------


def _ortho_poly(series: nw.Series[Any], degree: int) -> Frame:
    """R-style orthogonal polynomials via QR of a centered Vandermonde matrix.

    Reproduces R's :func:`stats::poly` on the non-null rows, then re-embeds
    the result into a full-length output so the row count matches the input.

    Parameters
    ----------
    series
        Single narwhals :class:`Series` of numeric values. Missing entries
        (narwhals nulls / NumPy NaNs) are excluded from the polynomial fit
        and become rows of ``NaN`` in the output — see Notes.
    degree
        Number of polynomial terms to return (the constant term is dropped).

    Returns
    -------
    Frame
        Narwhals frame with ``degree`` columns named ``{series.name}_1``
        through ``{series.name}_{degree}`` and the **same row count** as
        ``series``. The backend matches the input series.

    Notes
    -----
    Centering: we subtract the mean of the input before building the
    Vandermonde so that the higher-degree columns are orthogonal to the
    intercept (the constant column). Without centering, the QR step would
    mix them. After QR, multiplying ``Q`` by ``sign(diag(R))`` aligns the
    column signs with R's :func:`stats::poly`. The constant column is then
    dropped so only the polynomial terms remain.

    Missing values: the polynomial basis is fit on the non-null rows only
    (so a single NaN no longer poisons the global mean / Vandermonde / QR).
    The output array is then full-size and pre-filled with ``NaN``; the
    fitted basis values are written back into the non-null rows. This
    matches the per-column null handling done elsewhere in the package — a
    missing input gives a missing output for that row only.
    """
    # Coerce to a 1-D float array. The polynomial expansion is real-valued
    # regardless of whether the source dtype was integer or float. narwhals
    # nulls become NumPy NaN at this boundary.
    x = series.to_numpy().astype(float)

    # Per-row null mask. We fit the polynomial basis on the complete rows
    # only — otherwise even a single NaN would propagate through the mean,
    # the Vandermonde, and the QR factorization, producing an all-NaN result.
    mask = ~np.isnan(x)
    x_complete = x[mask]

    # Center the *complete* input. This is what makes the higher-degree
    # columns orthogonal to the intercept after the QR step.
    xc_complete = x_complete - x_complete.mean()

    # Vandermonde columns: [1, xc, xc^2, ..., xc^degree]. We cast to a concrete
    # ndarray type because numpy's `polyvander` is heavily overloaded and
    # pyright can't narrow the return type on its own. Mypy considers the cast
    # redundant (it sees a concrete type already), hence the targeted ignore.
    vander = cast(  # type: ignore[redundant-cast]
        "np.ndarray[Any, Any]", polyvander(xc_complete, degree)
    )

    # QR factorization: `q` has orthonormal columns, `r` is upper triangular.
    # Multiplying by sign(diag(R)) aligns the column signs with R's poly().
    q, r = np.linalg.qr(vander)
    diag_r = cast("np.ndarray[Any, Any]", np.diag(r))  # type: ignore[redundant-cast]
    z_complete = q * np.sign(diag_r)

    # Drop the constant (degree-0) column — we want only the new polynomial
    # terms, not an intercept.
    z_complete = z_complete[:, 1:]

    # Re-embed into a full-size, NaN-prefilled array so the output row count
    # matches the input. Rows that were null in the input stay NaN; rows that
    # were observed get their fitted basis values written back.
    z = np.full((x.shape[0], degree), np.nan)
    z[mask] = z_complete

    # 1-indexed names: x_1, x_2, ..., x_degree.
    columns = [f"{series.name}_{i}" for i in range(1, degree + 1)]

    # `series.implementation` carries the backend tag (pandas, polars, ...).
    # Building the new frame with the same backend keeps the recipe's
    # "pandas in, pandas out" / "polars in, polars out" guarantee.
    return nw.from_numpy(z, schema=columns, backend=series.implementation)


class StepPolynomial(Step):
    """Replace numeric columns with their orthogonal polynomial expansion.

    A ``ONE_TO_MANY`` step: each selected numeric column ``x`` is replaced in
    its positional slot by up to ``degree`` orthogonal polynomial columns
    ``{x}_1``, ``{x}_2``, ..., ``{x}_degree``, with the basis fitted on the
    training data. The math (R-style :func:`stats::poly` parity) lives in
    :func:`fit_polynomial` and :func:`apply_polynomial`; this class wraps them
    in the :class:`~shoez.step.Step` lifecycle so a
    :class:`~shoez.recipe.Recipe` can fit and apply it.

    The effective degree per column is capped at ``n_unique_non_null - 1`` — a
    column with ``k`` unique values can support at most ``k - 1`` orthogonal
    polynomial terms. Constant (or all-null) columns get an effective degree
    of ``0``; :meth:`transform` leaves those columns untouched and emits a
    :class:`UserWarning`. The caller can chain a zero-variance drop step to
    clean them up.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.

    Parameters
    ----------
    columns
        Names of the columns to expand.
    degree
        Maximum polynomial degree. Must be a positive integer. Each column's
        effective degree is the smaller of ``degree`` and
        ``n_unique_non_null - 1``.
    role
        Role assigned to the new polynomial columns. Default ``"predictor"``.
    id
        Stable step identifier. Default ``"polynomial"``.
    """

    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: list[str],
        *,
        degree: int = 2,
        role: str | None = "predictor",
        id: str | None = None,
    ) -> None:
        self.columns = list(columns)
        self.degree = degree
        self.id = id if id is not None else "polynomial"
        self.trained = False
        # ONE_TO_MANY replaces source columns with their expansion. There is
        # no apply-level toggle to keep the originals, so this stays False.
        self.keep_original_cols = False
        self.output_role = role
        # Populated by ``fit()``: the per-column effective degrees and schemas.
        self._fit_result: dict[str, int] | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Choose an effective polynomial degree for each selected column."""
        features = _frame.select(training_data, self.columns)
        self._fit_result = fit_polynomial(features, self.degree)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=list(self.columns), dtypes=dtypes_in)
        # Degree-0 columns stay put with their original dtype; the rest become
        # ``degree`` Float64 columns named {col}_1..{col}_d (same layout as
        # ``apply_polynomial`` produces).
        out_cols: list[str] = []
        out_dtypes: dict[str, str] = {}
        for col in self.columns:
            d = self._fit_result.get(col, 0)
            if d <= 0:
                out_cols.append(col)
                out_dtypes[col] = dtypes_in[col]
                continue
            for i in range(1, d + 1):
                name = f"{col}_{i}"
                out_cols.append(name)
                out_dtypes[name] = "Float64"
        self._schema_out = ColSchema(columns=out_cols, dtypes=out_dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Replace each selected column with its polynomial expansion."""
        if self._fit_result is None:
            raise RuntimeError("StepPolynomial.transform() called before fit().")
        return apply_polynomial(new_data, self._fit_result)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepPolynomial.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the polynomial columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepPolynomial.schema_out() called before fit().")
        return self._schema_out

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Orthogonal polynomial expansion is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Orthogonal polynomial bases are dense by construction."""
        return False
