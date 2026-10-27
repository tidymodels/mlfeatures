"""Internal dataframe boundary: the single data-ops API steps write against.

Per ``design/ecosystem.md``, narwhals is the *full internal representation* for
v0.1: a step is written once against this surface and narwhals dispatches to the
user's native backend (pandas in → pandas out, polars in → polars out). There are
no per-backend code paths.

This module wraps `narwhals.stable.v2` rather than re-exporting it, so the
escape-hatch discipline lives in one place: any operation that forces a drop to
the native frame via `to_native()` outside the fixed sklearn boundary is a
defect to upstream into narwhals, not a license to branch on backend.

``Frame`` is the internal type that flows between steps; ``to_native`` is called
only at the sklearn boundary (the output of ``Recipe.transform``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw
import numpy as np

from mlfeatures._dependencies import import_optional
from mlfeatures.selectors import FeaturesSelector
from mlfeatures.types import DataFrame, Frame, NwSelector, PlExpr, Roles, Selector

if TYPE_CHECKING:
    from mlfeatures.lineage import Lineage
    from mlfeatures.step import Step

# --- Entry / exit (the sklearn boundary) ----------------------------------


def from_native(df: DataFrame) -> Frame:
    """Wrap a native (pandas/polars/...) dataframe as an internal `Frame`.

    Rejects frames with duplicate column names up front. pandas allows repeated
    labels (polars does not), but every downstream operation here selects and
    aligns columns *by name*, so a duplicate makes the schema ambiguous. Failing
    at the boundary gives a clear, framework-owned error instead of a confusing
    failure (or silent wrong-column pick) deep inside a step.
    """
    try:
        return nw.from_native(df, eager_only=True)
    except nw.exceptions.DuplicateError as exc:
        # narwhals already rejects duplicate labels (which pandas permits but
        # polars does not), but its message is phrased in narwhals' terms.
        # Re-raise with a framework-owned message that names the offenders.
        dupes = _duplicate_names(df)
        listed = f": {dupes}" if dupes else ""
        raise ValueError(
            f"duplicate column names are not allowed{listed}; "
            "every column must have a unique name"
        ) from exc


def _duplicate_names(df: DataFrame) -> list[str]:
    """Names appearing more than once in ``df.columns``, in first-seen order.

    Reads ``columns`` off the native frame (both pandas and polars expose it);
    used only to enrich the duplicate-column error, so an unexpected shape just
    yields an empty list rather than masking the original failure.
    """
    columns = getattr(df, "columns", None)
    if columns is None:
        return []
    seen: set[str] = set()
    dupes: list[str] = []
    for name in list(columns):
        if name in seen and name not in dupes:
            dupes.append(name)
        seen.add(name)
    return dupes


def to_native(frame: Frame) -> DataFrame:
    """Unwrap an internal `Frame` back to its native dataframe.

    Called only at the fixed sklearn boundary (the output of ``transform``).
    """
    return frame.to_native()


# --- Column selection ------------------------------------------------------


def columns(frame: Frame) -> list[str]:
    """Column names, in order."""
    return frame.columns


def select(
    frame: Frame,
    names: Selector,
    roles: Roles | None = None,
    lineage: "Lineage | None" = None,
    steps: "dict[str, Step] | None" = None,
) -> Frame:
    """Return a frame with only the selected columns.

    ``names`` may be a single column name, a sequence of names (kept in the
    given order), a narwhals selector (e.g. ``narwhals.selectors.numeric()``),
    a polars selector (``polars.selectors``) or column-selection expression
    (``polars.col("a")``), or a `FeaturesSelector`. A polars selector or
    expression is resolved against the frame's schema, so it works
    regardless of the frame's backend. A ``FeaturesSelector`` additionally uses
    ``roles`` to filter by role, and (for provenance selectors) the live
    lineage DAG ``lineage`` and id→step map ``steps`` when the recipe supplies
    them at fit time.
    """
    if isinstance(names, FeaturesSelector):
        return frame.select(names.resolve(frame, roles or Roles(), lineage, steps))
    if isinstance(names, str):
        return frame.select([names])
    if isinstance(names, NwSelector):
        return frame.select(names)
    # The switch into polars handling: a polars selector is an ``Expr`` subclass
    # on every supported version, so one check covers both selectors
    # (``polars.selectors.numeric()``) and plain column selections
    # (``polars.col("a")``). ``PlExpr`` consults polars only if it is already
    # loaded, since a polars expression cannot exist otherwise.
    if isinstance(names, PlExpr):
        return frame.select(_expand_polars_expr(frame, names))
    # Only a sequence of names is left.
    return frame.select(list(names))


def _expand_polars_expr(frame: Frame, expr: PlExpr) -> list[str]:
    """Resolve a polars selector or column selection to names against ``frame``.

    Polars expressions are not understood by narwhals, so the expression is
    expanded by ``polars.selectors.expand_selector`` against the frame's schema
    (via `narwhals.Schema.to_polars()`). This is pure name resolution that
    touches no data and works for any backend (the schema is normalized by
    narwhals).

    ``strict=False`` is what admits a plain column selection such as
    ``polars.col("a")`` alongside true selectors; polars still refuses an
    expression that computes or renames (``polars.col("a") + 1``,
    ``.alias()``), which has no meaning as a selection. It is also what makes
    this safe for a selector that has been through ``copy.deepcopy``, as every
    step's is in `Recipe.fit`: before polars 1.32 the strict check relies on a
    private attribute that a deep copy drops.
    """
    # Reached only when ``names`` is a polars expression, which it can be
    # only if polars is loaded, so polars is guaranteed importable here and
    # this never raises. Routed through import_optional anyway so the package's
    # sanctioned runtime-backend imports are uniform/greppable.
    cs = import_optional("polars.selectors", purpose="to resolve a polars selector")

    try:
        return list(cs.expand_selector(frame.schema.to_polars(), expr, strict=False))
    except TypeError as exc:
        raise TypeError(
            f"polars expression {str(expr)!r} is not a column selection; "
            "pass a selection such as pl.col('a') or a polars selector"
        ) from exc


def drop(frame: Frame, names: str | Sequence[str]) -> Frame:
    """Return a frame with ``names`` removed."""
    cols = [names] if isinstance(names, str) else list(names)
    return frame.drop(cols)


def with_column(frame: Frame, name: str, values: Any) -> Frame:
    """Return a frame with ``values`` attached as a column named ``name``.

    Used at the sklearn boundary to re-join a target ``y`` (a numpy array, list, or
    Series) onto the feature frame under its target-role name. ``values`` is wrapped as a
    narwhals `Series` on the frame's backend so the result stays native (pandas in →
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
    `narwhals.Categorical` or `narwhals.Enum`.
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


def is_ordered_categorical(frame: Frame, column: str) -> bool:
    """Whether ``column`` is an *ordered* categorical.

    narwhals normalizes ordered categoricals to `narwhals.Enum` (polars ``Enum``
    and pandas ``Categorical(ordered=True)``) and unordered ones to
    `narwhals.Categorical` (polars ``Categorical`` and pandas
    ``Categorical(ordered=False)``). So ordering *is* preserved across backends:
    ``Enum`` is ordered, ``Categorical`` is not.
    """
    return frame.schema[column] == nw.Enum


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


def is_duration(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a Duration (time-delta) dtype."""
    return isinstance(frame.schema[column], nw.Duration)


def is_list(frame: Frame, column: str) -> bool:
    """Whether ``column`` has a List dtype (a variable-length list per row)."""
    return isinstance(frame.schema[column], nw.List)


# --- Taxonomy classification -----------------------------------------------


def dtype_category(dtype: nw.dtypes.DType) -> str:
    """Classify a narwhals dtype into a `design/lineage.md` taxonomy category.

    Returns one of the taxonomy leaf names: ``"float"``, ``"int"``,
    ``"decimal"``, ``"bool"``, ``"str"``, ``"categorical_ordered"``,
    ``"categorical_unordered"``, ``"date"``, ``"datetime"``, ``"duration"``,
    or ``"list"``. Anything narwhals reports that falls outside the taxonomy
    maps to ``"other"``.

    This is the single place that names the taxonomy categories; the lineage
    DAG and ``recipe.summary()`` (see ``design/lineage.md``) will consume it.
    ``"decimal"`` is split out from the numeric group because narwhals'
    ``Decimal`` is neither ``is_float`` nor ``is_integer``, so it has no home
    under the float/int leaves of the taxonomy tree.
    """
    if dtype.is_float():
        return "float"
    if dtype.is_integer():
        return "int"
    if dtype == nw.Decimal:
        return "decimal"
    if dtype == nw.Boolean:
        return "bool"
    if dtype == nw.String:
        return "str"
    if dtype == nw.Enum:
        return "categorical_ordered"
    if dtype == nw.Categorical:
        return "categorical_unordered"
    if dtype == nw.Date:
        return "date"
    if isinstance(dtype, nw.Datetime):
        return "datetime"
    if isinstance(dtype, nw.Duration):
        return "duration"
    if isinstance(dtype, nw.List):
        return "list"
    return "other"


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


# --- NumPy bridge ----------------------------------------------------------


def _is_numpy_unrepresentable(dtype: nw.dtypes.DType) -> bool:
    """Whether ``dtype`` has no NumPy representation in polars' bridge.

    128-bit integers have no NumPy scalar type, so a bare ``to_numpy()`` on such
    a column raises a Rust ``PanicException`` rather than a clean error. Callers
    upcast these to ``Float64`` (which every numpy-backed step coerces to anyway)
    before crossing the boundary. See issue #108.
    """
    return dtype == nw.Int128 or dtype == nw.UInt128


def _cast_numpy_safe(frame: Frame) -> Frame:
    """Upcast any ``Int128``/``UInt128`` columns to ``Float64`` in place."""
    casts = [
        nw.col(name).cast(nw.Float64)
        for name, dtype in frame.schema.items()
        if _is_numpy_unrepresentable(dtype)
    ]
    return frame.with_columns(*casts) if casts else frame


def to_float_array(frame: Frame) -> "np.ndarray[Any, Any]":
    """Convert ``frame`` to a ``float64`` NumPy array without panicking.

    Columns whose dtype has no NumPy representation (``Int128``/``UInt128``) are
    upcast to ``Float64`` first, so steps that route data through NumPy can
    honestly accept every numeric dtype in `NUMERIC_DTYPES`.
    """
    return _cast_numpy_safe(frame).to_numpy().astype(float)


def series_to_float_array(series: "nw.Series[Any]") -> "np.ndarray[Any, Any]":
    """Convert a `Series` to a ``float64`` NumPy array without panicking.

    The single-column analogue of `to_float_array`; see issue #108.
    """
    if _is_numpy_unrepresentable(series.dtype):
        series = series.cast(nw.Float64)
    return series.to_numpy().astype(float)


# --- Scalar extraction -----------------------------------------------------


def as_float(value: Any) -> float:
    """Coerce a narwhals scalar to ``float``, mapping a missing value to NaN.

    Aggregations such as ``Series.mean`` / ``Series.std`` return ``None`` when
    they are undefined (e.g. an empty column); callers storing the result as a
    numeric statistic want ``NaN`` to propagate rather than a ``None``.
    """
    return float("nan") if value is None else float(value)
