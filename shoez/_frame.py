"""Internal dataframe boundary — the single data-ops API steps write against.

Per ``design/ecosystem.md``, narwhals is the *full internal representation* for
v0.1: a step is written once against this surface and narwhals dispatches to the
user's native backend (pandas in → pandas out, polars in → polars out). There are
no per-backend code paths.

This module wraps :mod:`narwhals.stable.v2` rather than re-exporting it, so the
escape-hatch discipline lives in one place: any operation that forces a drop to
the native frame via :func:`to_native` outside the fixed sklearn boundary is a
defect to upstream into narwhals, not a license to branch on backend.

``Frame`` is the internal type that flows between steps; ``to_native`` is called
only at the sklearn boundary (the output of ``Recipe.transform``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import narwhals.stable.v2 as nw

from shoez.selectors import ShoezSelector
from shoez.types import DataFrame, Frame, NwSelector, PlSelector, Roles, Selector
from shoez._dependencies import import_optional

# --- Entry / exit (the sklearn boundary) ----------------------------------


def from_native(df: DataFrame) -> Frame:
    """Wrap a native (pandas/polars/...) dataframe as an internal :data:`Frame`."""
    return nw.from_native(df, eager_only=True)


def to_native(frame: Frame) -> DataFrame:
    """Unwrap an internal :data:`Frame` back to its native dataframe.

    Called only at the fixed sklearn boundary (the output of ``transform``).
    """
    return frame.to_native()


# --- Column selection ------------------------------------------------------


def columns(frame: Frame) -> list[str]:
    """Column names, in order."""
    return frame.columns


def select(frame: Frame, names: Selector, roles: Roles | None = None) -> Frame:
    """Return a frame with only the selected columns.

    ``names`` may be a single column name, a sequence of names (kept in the
    given order), a narwhals selector (e.g. ``narwhals.selectors.numeric()``),
    a polars selector (``polars.selectors``), or a :class:`~shoez.selectors.ShoezSelector`.
    A polars selector is resolved against the frame's schema, so it works
    regardless of the frame's backend. A ``ShoezSelector`` additionally uses
    ``roles`` to filter by role.
    """
    if isinstance(names, ShoezSelector):
        return frame.select(names.resolve(frame, roles or Roles()))
    if isinstance(names, str):
        return frame.select([names])
    if isinstance(names, NwSelector):
        return frame.select(names)
    if isinstance(names, PlSelector):
        return frame.select(_expand_polars_selector(frame, names))
    return frame.select(list(names))


def _expand_polars_selector(frame: Frame, selector: PlSelector) -> list[str]:
    """Resolve a polars selector to concrete column names against ``frame``.

    Polars selectors are not understood by narwhals, so the selector is expanded
    against the frame's schema converted to a polars schema via
    :meth:`narwhals.Schema.to_polars`. This is pure name resolution that touches
    no data and works for any backend (the schema is normalized by narwhals).
    """
    # Reached only when ``names`` is already a PlSelector instance, which the
    # databackend hook matches only if polars is loaded — so polars is guaranteed
    # importable here and this never raises. Routed through import_optional anyway
    # so the package's sanctioned runtime-backend imports are uniform/greppable.
    cs = import_optional("polars.selectors", purpose="to resolve a polars selector")

    return list(cs.expand_selector(frame.schema.to_polars(), selector))


def drop(frame: Frame, names: str | Sequence[str]) -> Frame:
    """Return a frame with ``names`` removed."""
    cols = [names] if isinstance(names, str) else list(names)
    return frame.drop(cols)


def with_column(frame: Frame, name: str, values: Any) -> Frame:
    """Return a frame with ``values`` attached as a column named ``name``.

    Used at the sklearn boundary to re-join a target ``y`` (a numpy array, list, or
    Series) onto the feature frame under its outcome-role name. ``values`` is wrapped as a
    narwhals :class:`Series` on the frame's backend so the result stays native (pandas in →
    pandas out).
    """
    series = nw.new_series(name=name, values=values, backend=frame.implementation)
    return frame.with_columns(series.alias(name))


def apply_drop(frame: Frame, drop_cols: list[str]) -> Frame:
    """Return ``frame`` with ``drop_cols`` removed, silently skipping absent names.

    The column-dropping half of every REMOVING-mode step (``step_drop_zv``,
    ``step_drop_corr``, ...). Each fitter inspects training data and returns
    *which* columns to remove; this helper performs the removal. Splitting the
    work means the same names can be applied to both the training frame and
    any held-out frame, guaranteeing train/serve parity (the same columns
    disappear at fit and at transform time).

    Names not present in ``frame`` are skipped rather than raising, which
    matches the framework's policy of tolerating schema drift at transform
    time (see ``design/lifecycle.md``).
    """
    present = [c for c in drop_cols if c in frame.columns]
    return drop(frame, present)


# --- Type-aware ops --------------------------------------------------------


def dtypes(frame: Frame) -> dict[str, nw.dtypes.DType]:
    """Map of column name to its narwhals dtype."""
    return dict(frame.schema)


def is_numeric(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a numeric (int/float) dtype."""
    return frame.schema[column].is_numeric()


def numeric_columns(frame: Frame) -> list[str]:
    """Names of the numeric columns, in order."""
    return [name for name, dtype in frame.schema.items() if dtype.is_numeric()]


# --- Categorical handling --------------------------------------------------


def is_categorical(frame: Frame, column: str) -> bool:
    """Whether ``column`` is categorical or enum-typed.

    Covers ``pandas.Categorical`` (ordered and unordered) and polars
    ``Categorical`` / ``Enum``, all of which narwhals normalizes to
    :class:`narwhals.Categorical` or :class:`narwhals.Enum`.
    """
    dtype = frame.schema[column]
    return dtype == nw.Categorical or dtype == nw.Enum


def categorical_columns(frame: Frame) -> list[str]:
    """Names of the categorical/enum columns, in order."""
    return [
        name
        for name, dtype in frame.schema.items()
        if dtype == nw.Categorical or dtype == nw.Enum
    ]


def is_string(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a string dtype."""
    return frame.schema[column] == nw.String


def is_integer(frame: Frame, column: str) -> bool:
    """Whether ``column`` has an integer dtype (signed or unsigned)."""
    return frame.schema[column].is_integer()


def is_double(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a floating-point dtype."""
    return frame.schema[column].is_float()


def is_date(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a Date dtype."""
    return frame.schema[column] == nw.Date


def is_datetime(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a Datetime dtype."""
    return isinstance(frame.schema[column], nw.Datetime)


def is_boolean(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a Boolean dtype."""
    return frame.schema[column] == nw.Boolean


def is_nominal(frame: Frame, column: str) -> bool:
    """Whether ``column`` is nominal: categorical, enum, or string."""
    dtype = frame.schema[column]
    return dtype == nw.String or dtype == nw.Categorical or dtype == nw.Enum


# --- Missing-value awareness -----------------------------------------------


def null_counts(frame: Frame) -> dict[str, int]:
    """Map of column name to its count of null values.

    Treats NumPy ``NaN`` and pandas/Arrow nullable nulls uniformly, as narwhals
    normalizes both to the null concept.
    """
    counts = frame.null_count()
    return {name: int(counts[name][0]) for name in counts.columns}


def has_nulls(frame: Frame, column: str) -> bool:
    """Whether ``column`` contains any null values."""
    return frame[column].null_count() > 0


# --- Scalar extraction -----------------------------------------------------


def as_float(value: Any) -> float:
    """Coerce a narwhals scalar to ``float``, mapping a missing value to NaN.

    Aggregations such as ``Series.mean`` / ``Series.std`` return ``None`` when
    they are undefined (e.g. an empty column); callers storing the result as a
    numeric statistic want ``NaN`` to propagate rather than a ``None``.
    """
    return float("nan") if value is None else float(value)
