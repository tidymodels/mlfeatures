"""Internal dataframe boundary — the single data-ops API steps write against.

Per ``design/ecosystem.md``, narwhals is the *full internal representation* for
v0.1: a step is written once against this surface and narwhals dispatches to the
user's native backend (pandas in → pandas out, polars in → polars out). There are
no per-backend code paths.

This module wraps :mod:`narwhals.stable.v1` rather than re-exporting it, so the
escape-hatch discipline lives in one place: any operation that forces a drop to
the native frame via :func:`to_native` outside the fixed sklearn boundary is a
defect to upstream into narwhals, not a license to branch on backend.

``Frame`` is the internal type that flows between steps; ``to_native`` is called
only at the sklearn boundary (the output of ``Recipe.transform``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias

import narwhals.stable.v1 as nw

from shoez.types import DataFrame

#: The internal frame type that flows between steps. A backend-agnostic narwhals
#: ``DataFrame`` wrapping the user's native (pandas/polars/...) frame.
Frame: TypeAlias = nw.DataFrame[Any]


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


def select(frame: Frame, names: str | Sequence[str]) -> Frame:
    """Return a frame with only ``names`` (in the given order)."""
    cols = [names] if isinstance(names, str) else list(names)
    return frame.select(cols)


def drop(frame: Frame, names: str | Sequence[str]) -> Frame:
    """Return a frame with ``names`` removed."""
    cols = [names] if isinstance(names, str) else list(names)
    return frame.drop(cols)


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
