"""Shared spline-basis machinery for `StepSplineNatural` and `StepSplineMonotone`.

Ported from R `splines2`'s Armadillo header-only C++ core
(``splines2Armadillo/{NaturalSpline,BSpline,MSpline,ISpline}.h``), which R
``recipes``' ``step_spline_natural()`` / ``step_spline_monotone()`` wrap. This
module holds the pieces genuinely shared by both step families -- the
B-spline design-matrix primitive, quantile-based knot placement, and common
validation -- mirroring R's own ``spline_helpers.R``. The per-family basis
math (the natural-spline null-space transform, the M-spline-to-I-spline
antiderivative identity) lives in each step's own module, since it is not
shared.

Not part of the public API: only `mlfeatures.step_spline_natural` and
`mlfeatures.step_spline_monotone` import from here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from mlfeatures._dependencies import import_optional

__all__: list[str] = []


def padded_knot_vector(
    internal_knots: "np.ndarray[Any, Any]",
    boundary_knots: tuple[float, float],
    degree: int,
) -> "np.ndarray[Any, Any]":
    """The full de Boor knot vector: boundary knots repeated ``degree + 1`` times.

    Standard B-spline padding: an order-``degree + 1`` B-spline needs each
    boundary knot repeated ``degree + 1`` times so the basis is anchored (not
    just tangent) at the boundary. ``internal_knots`` sit strictly between.
    """
    order = degree + 1
    return np.concatenate(
        (
            np.full(order, boundary_knots[0]),
            np.asarray(internal_knots, dtype=float),
            np.full(order, boundary_knots[1]),
        )
    )


def bspline_design_matrix(
    x: "np.ndarray[Any, Any]",
    internal_knots: "np.ndarray[Any, Any]",
    boundary_knots: tuple[float, float],
    degree: int,
    deriv: int = 0,
) -> "np.ndarray[Any, Any]":
    """Standard order-``degree + 1`` B-spline design matrix, evaluated at ``x``.

    Column ``i`` is the ``i``-th one-hot B-spline basis function (or its
    ``deriv``-th derivative), evaluated with polynomial-continuation
    extrapolation past the boundary (SciPy's ``extrapolate=True``). Callers
    needing a *different* extrapolation rule (natural splines extrapolate
    linearly, not as the raw local cubic) evaluate this at the boundary knot
    itself and build the extrapolated row by hand -- see
    `mlfeatures.step_spline_natural`.
    """
    bspline_cls = import_optional(
        "scipy.interpolate", purpose="for spline basis construction"
    ).BSpline
    t = padded_knot_vector(internal_knots, boundary_knots, degree)
    order = degree + 1
    n_bases = len(t) - order
    design = np.empty((len(x), n_bases))
    for i in range(n_bases):
        coef = np.zeros(n_bases)
        coef[i] = 1.0
        spline = bspline_cls(t, coef, degree, extrapolate=True)
        if deriv:
            spline = spline.derivative(deriv)
        design[:, i] = spline(x)
    return design


def quantile_interior_knots(
    x: "np.ndarray[Any, Any]", n_interior: int
) -> "np.ndarray[Any, Any]":
    """Place ``n_interior`` interior knots at evenly-spaced quantiles of ``x``.

    Matches R's default knot placement (``stats::quantile(x, probs =
    (1:n)/(n+1))``): NumPy's default ``"linear"`` interpolation is the same as
    R's quantile type 7, verified against live R `splines2` output.
    """
    if n_interior <= 0:
        return np.array([], dtype=float)
    probs = np.arange(1, n_interior + 1) / (n_interior + 1)
    return cast("np.ndarray[Any, Any]", np.quantile(x, probs))


def capped_interior_knots(
    x: "np.ndarray[Any, Any]", n_interior: int
) -> "np.ndarray[Any, Any]":
    """Quantile-placed interior knots, deduplicated.

    Low-cardinality training data can make requested quantile knots collide
    (two knots landing on the same repeated value). Rather than let the
    downstream B-spline construction choke on a degenerate knot sequence --
    R `splines2`'s behavior, via ``spline2_create()``'s try/catch, is to drop
    the whole column -- mlfeatures shrinks the interior knot count to whatever came
    out unique. This is what makes the natural/monotone fits "cap silently"
    rather than "drop the column", matching `StepPolynomial`'s established
    house style (see ``fit_polynomial``'s degree-capping).
    """
    return cast(
        "np.ndarray[Any, Any]", np.unique(quantile_interior_knots(x, n_interior))
    )


def boundary_knots(x: "np.ndarray[Any, Any]") -> tuple[float, float]:
    """The ``(min, max)`` of ``x``, used as the boundary knots."""
    return float(np.min(x)), float(np.max(x))


def check_deg_free(deg_free: int, *, minimum: int, arg: str = "deg_free") -> None:
    """Validate ``deg_free`` is an ``int`` (not ``bool``) at least ``minimum``."""
    if (
        isinstance(deg_free, bool)
        or not isinstance(deg_free, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or deg_free < minimum
    ):
        raise ValueError(
            f"'{arg}' must be an integer greater than or equal to {minimum}, "
            f"got {deg_free!r}"
        )


def check_degree(degree: int, *, arg: str = "degree") -> None:
    """Validate ``degree`` is a non-negative ``int`` (not ``bool``)."""
    if (
        isinstance(degree, bool)
        or not isinstance(degree, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or degree < 0
    ):
        raise ValueError(
            f"'{arg}' must be an integer greater than or equal to 0, got {degree!r}"
        )


def check_complete_set(complete_set: bool, *, arg: str = "complete_set") -> None:
    """Validate ``complete_set`` is a plain ``bool``."""
    if not isinstance(complete_set, bool):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError(f"'{arg}' must be a bool, got {complete_set!r}")


def check_explicit_knots(
    knots: Sequence[float],
    boundary: tuple[float, float],
    *,
    column: str,
    arg: str = "knots",
) -> "np.ndarray[Any, Any]":
    """Validate and sort caller-supplied interior ``knots`` for one column.

    Returns the sorted array of interior knots. Unlike the quantile-placed
    default, explicit knots are the caller's responsibility -- a duplicate or
    out-of-boundary value is a configuration error (`ValueError` naming the
    offending column), not something the step caps and warns about.
    """
    arr = np.sort(np.asarray(list(knots), dtype=float))
    if arr.size != len(set(arr.tolist())):
        raise ValueError(
            f"'{arg}' must not contain duplicate values, got {list(knots)!r} "
            f"for column {column!r}"
        )
    lo, hi = boundary
    if arr.size and (arr[0] <= lo or arr[-1] >= hi):
        raise ValueError(
            f"'{arg}' must lie strictly inside the boundary knots "
            f"({lo!r}, {hi!r}) for column {column!r}, got {list(knots)!r}"
        )
    return arr
