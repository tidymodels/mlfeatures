"""Shared calendar-part-expansion machinery for `StepDateFeatures` and
`StepDatetimeFeatures`.

The two steps are deliberately separate classes -- a date has no time of day,
mirroring Python's own ``datetime.date`` / ``datetime.datetime`` split (see
each step module's own docstring for the full rationale) -- but the mechanics
of validating a requested feature list, normalizing it into a canonical order,
building the ``categorical``-vs-plain-integer expression for a given feature,
and replacing each selected column with its calendar-part columns are
identical between them. This module holds that shared logic, parameterized by
each step's own feature vocabulary (the ``{feature: expr}`` mapping each step
module defines for itself); each step module keeps its own public
`fit_date_features()` / `apply_date_features()` etc. with their own
docstrings, delegating to the functions here.

Not part of the public API: only `mlfeatures.step_date_features` and
`mlfeatures.step_datetime_features` import from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures.types import Frame

if TYPE_CHECKING:
    from collections.abc import Callable


def year_expr(e: nw.Expr) -> nw.Expr:
    return e.dt.year()


def month_expr(e: nw.Expr) -> nw.Expr:
    return e.dt.month()


def day_expr(e: nw.Expr) -> nw.Expr:
    return e.dt.day()


def dow_expr(e: nw.Expr) -> nw.Expr:
    return e.dt.weekday()


def doy_expr(e: nw.Expr) -> nw.Expr:
    return e.dt.ordinal_day()


#: Ordered level labels for the two features that name a natural, fixed
#: cycle. Every other feature (``year``, ``day``, ``doy``, and, for
#: Datetime, ``hour``/``minute``/``second``) has no such vocabulary -- there
#: is no name for "second 47" -- so ``month`` and ``dow`` are the only
#: features ``categorical`` accepts, for both steps alike.
MONTH_LABELS: list[str] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
DOW_LABELS: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
CATEGORICAL_LABELS: dict[str, list[str]] = {"month": MONTH_LABELS, "dow": DOW_LABELS}
#: Order matches each step's own default-features order (``month`` before
#: ``dow``), so normalizing ``categorical`` against this tuple keeps it
#: consistent with ``features``' own canonical order.
CATEGORICAL_ELIGIBLE: tuple[str, ...] = ("month", "dow")


def validate_features(
    features: tuple[str, ...], exprs: dict[str, Callable[[nw.Expr], nw.Expr]]
) -> None:
    """Raise unless every entry of ``features`` is a key of ``exprs``."""
    invalid = [f for f in features if f not in exprs]
    if invalid:
        raise ValueError(
            f"'features' must be a subset of {sorted(exprs)!r}, got {invalid!r}"
        )


def normalize_features(
    features: tuple[str, ...], default_order: tuple[str, ...]
) -> tuple[str, ...]:
    """Dedup ``features`` and order it canonically (``default_order``'s order).

    Two constructor calls differing only in the order or duplication of the
    same requested parts should behave identically -- same output columns,
    same ``id`` collision behavior, same fitted state -- rather than produce
    columns in whatever order the caller happened to type them.
    """
    requested = set(features)
    return tuple(f for f in default_order if f in requested)


def validate_categorical(
    categorical: tuple[str, ...], features: tuple[str, ...]
) -> None:
    """Raise unless ``categorical`` is eligible and was also requested in ``features``."""
    invalid = [f for f in categorical if f not in CATEGORICAL_ELIGIBLE]
    if invalid:
        raise ValueError(
            f"'categorical' must be a subset of {sorted(CATEGORICAL_ELIGIBLE)!r}, "
            f"got {invalid!r}"
        )
    not_requested = [f for f in categorical if f not in features]
    if not_requested:
        raise ValueError(
            f"'categorical' entries must also be requested in 'features'; "
            f"not requested: {not_requested!r}"
        )


def normalize_categorical(categorical: tuple[str, ...]) -> tuple[str, ...]:
    """Dedup ``categorical`` and order it canonically (`CATEGORICAL_ELIGIBLE`'s order)."""
    requested = set(categorical)
    return tuple(f for f in CATEGORICAL_ELIGIBLE if f in requested)


def feature_expr(
    col_name: str,
    feature: str,
    exprs: dict[str, Callable[[nw.Expr], nw.Expr]],
    *,
    categorical: bool,
) -> nw.Expr:
    """Build the expression computing ``feature`` from ``col_name``, aliased.

    Plain features are cast to ``Int32`` for backend parity: polars picks the
    narrowest integer width per part while pandas' narwhals backend always
    reports ``Int32``. A ``categorical`` feature is instead mapped to its
    ordered level labels (`CATEGORICAL_LABELS`) and cast to a narwhals
    ``Enum`` of those labels, so ``month``/``dow`` come back with readable
    level names (``"January"``, ``"Monday"``, ...) instead of raw numbers.
    """
    name = f"{col_name}_{feature}"
    raw = exprs[feature](nw.col(col_name))
    if categorical:
        labels = CATEGORICAL_LABELS[feature]
        mapping = {i + 1: label for i, label in enumerate(labels)}
        return raw.replace_strict(mapping, return_dtype=nw.Enum(labels)).alias(name)
    return raw.cast(nw.Int32).alias(name)


def output_dtype_name(feature: str, *, categorical: tuple[str, ...]) -> str:
    """The dtype string `feature_expr()` produces for ``feature``, for `ColSchema`."""
    if feature in categorical:
        return str(nw.Enum(CATEGORICAL_LABELS[feature]))
    return "Int32"


def apply_calendar_features(
    data: Frame,
    columns: list[str],
    features: tuple[str, ...],
    categorical: tuple[str, ...],
    exprs: dict[str, Callable[[nw.Expr], nw.Expr]],
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Replace each column in ``columns`` with its requested calendar-part columns.

    For every column in ``columns``, adds one column per entry of
    ``features``, named ``{column}_{feature}`` and placed in the column's
    original slot, replacing it (or following it, kept, under
    ``keep_original_cols``). A feature named in ``categorical`` is encoded as
    an ordered ``Enum`` of level names instead of a plain integer. Every other
    column of ``data`` passes through unchanged.
    """
    final_order: list[str] = []
    to_drop: list[str] = []
    new_cols: list[nw.Expr] = []
    selected_cols = set(columns)
    categorical_set = set(categorical)
    for col_name in data.columns:
        if col_name not in selected_cols:
            final_order.append(col_name)
            continue
        if keep_original_cols:
            final_order.append(col_name)
        else:
            to_drop.append(col_name)
        for feature in features:
            final_order.append(f"{col_name}_{feature}")
            new_cols.append(
                feature_expr(
                    col_name, feature, exprs, categorical=feature in categorical_set
                )
            )

    if new_cols:
        data = data.with_columns(*new_cols)
    if to_drop:
        data = data.drop(to_drop)
    return data.select(final_order)
