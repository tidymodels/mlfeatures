"""Minimal formula parser — sugar over explicit role assignment.

Per ``design/lifecycle.md``, the formula interface is intentionally tiny: it
supports only simple outcome/predictor splits. ``"y ~ ."`` makes every other
column a predictor; ``"y ~ x1 + x2"`` names predictors explicitly. There are no
interactions, transformations, or arithmetic — anything beyond plain column
selection must be expressed as a step. This keeps the parser well clear of
Patsy-style complexity.
"""

from __future__ import annotations

import re

from shoez.types import Roles

#: Characters that signal formula features we deliberately do not support
#: (interactions, arithmetic, transformations, grouping). Their presence is a
#: hard error pointing the user at steps instead.
_UNSUPPORTED = re.compile(r"[*:()\[\]^/%-]|\bI\b")


def parse_formula(formula: str, columns: list[str]) -> Roles:
    """Parse ``formula`` into role assignments against ``columns``.

    Only ``lhs ~ rhs`` with ``+``-separated names (or ``.`` on the RHS for "all
    remaining columns") is accepted. Raises :class:`ValueError` on anything
    richer, on unknown column names, or on a malformed expression.
    """
    if formula.count("~") != 1:
        raise ValueError(f"formula must contain exactly one '~': got {formula!r}")
    lhs, rhs = (side.strip() for side in formula.split("~"))

    for side in (lhs, rhs):
        if _UNSUPPORTED.search(side):
            raise ValueError(
                f"unsupported formula syntax in {formula!r}: only simple "
                "'y ~ .' or 'y ~ x1 + x2' splits are allowed — express "
                "interactions, transformations, or arithmetic as steps instead"
            )

    outcomes = _parse_terms(lhs, formula)
    if rhs == ".":
        predictors = [c for c in columns if c not in outcomes]
    else:
        predictors = _parse_terms(rhs, formula)

    known = set(columns)
    for name in (*outcomes, *predictors):
        if name not in known:
            raise ValueError(f"formula references unknown column {name!r}")

    roles = Roles()
    for name in outcomes:
        roles.add(name, "outcome")
    for name in predictors:
        roles.add(name, "predictor")
    return roles


def _parse_terms(side: str, formula: str) -> list[str]:
    """Split a ``+``-separated side into trimmed, validated column names."""
    if not side:
        raise ValueError(f"formula side is empty in {formula!r}")
    terms = [term.strip() for term in side.split("+")]
    if any(not term for term in terms):
        raise ValueError(f"malformed formula term in {formula!r}")
    return terms
