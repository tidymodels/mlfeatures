"""Shared fit-schema and role-update logic for the basis-expansion steps.

`StepPolynomial`, `StepSplineNatural`, and `StepSplineMonotone` all replace
each selected column with ``degree``/``deg_free`` new ``{col}_1``..``{col}_d``
``Float64`` columns, and leave a column whose effective degree is ``0``
(too few unique values to support any basis term) untouched -- a
pass-through, not a newly produced column, so it is excluded from
``output_parents``. `fit()`'s output-schema construction and
``update_roles()`` follow purely from that shape, with nothing step-specific
left once each step supplies its own per-column fit object and how to read
its degree/deg_free off it. This module holds that shared logic.

Not part of the public API: only ``mlfeatures.step_polynomial``,
``mlfeatures.step_spline_natural``, and ``mlfeatures.step_spline_monotone`` import
from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from mlfeatures.types import ColSchema

if TYPE_CHECKING:
    from collections.abc import Callable

    from mlfeatures.types import Roles

FitT = TypeVar("FitT")


def expand_columns_schema(
    schema_in: ColSchema,
    estimates: dict[str, FitT],
    degree_of: "Callable[[FitT], int]",
) -> tuple[ColSchema, dict[str, list[str]]]:
    """The ``(schema_out, output_parents)`` pair for a basis-expansion step's ``fit()``.

    For each column in ``schema_in``, ``degree_of(estimates[col])`` (or ``0``
    if the column has no entry in ``estimates``) decides its fate: a degree of
    ``0`` passes the column through unchanged, keeping its input dtype; a
    positive degree ``d`` produces ``d`` new ``Float64`` columns named
    ``{col}_1``..``{col}_d``, each descending from ``col`` in the returned
    ``output_parents`` map.
    """
    out_cols: list[str] = []
    out_dtypes: dict[str, str] = {}
    parents: dict[str, list[str]] = {}
    for col in schema_in.columns:
        fit = estimates.get(col)
        degree = degree_of(fit) if fit is not None else 0
        if degree <= 0:
            out_cols.append(col)
            out_dtypes[col] = schema_in.dtypes[col]
            continue
        for i in range(1, degree + 1):
            name = f"{col}_{i}"
            out_cols.append(name)
            out_dtypes[name] = "Float64"
            parents[name] = [col]
    return ColSchema(columns=out_cols, dtypes=out_dtypes), parents


def update_roles_for_expansion(
    roles: "Roles",
    output_parents: dict[str, list[str]],
    *,
    output_role: str | None,
    keep_original_cols: bool,
) -> None:
    """Propagate roles for a basis-expansion step's produced columns.

    Every produced column gets ``output_role`` (if any); unless
    ``keep_original_cols``, every expanded source column (one that actually
    produced output, i.e. excluding a degree-0 pass-through) loses its
    existing roles.
    """
    if output_role is not None:
        for col in output_parents:
            roles.add(col, output_role)
    if keep_original_cols:
        return
    expanded = {parent for parents in output_parents.values() for parent in parents}
    for col in expanded:
        for role in list(roles.of(col)):
            roles.remove(col, role)
