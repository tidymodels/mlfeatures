"""Calendar-part expansion for Date columns: fit + apply.

`step_date_features` is the Date-only counterpart of
`step_datetime_features` (`shoez/step_datetime_features.py`). They are two
separate steps because a date has no time of day: ``hour``, ``minute``, and
``second`` make no sense on a plain calendar date, so a single step trying to
cover both would need to either fake a time-of-day or special-case Date
columns internally. Splitting mirrors Python's own ``datetime.date`` /
``datetime.datetime`` split (``datetime`` does not inherit from ``date``).

The *fit* half (`fit_date_features()`) only needs the selected columns' names
and the requested feature list; there is nothing to estimate from the data.
The *apply* half (`apply_date_features()`) computes each requested calendar
part. `StepDateFeatures` wraps both into the `Step` lifecycle so a `Recipe`
can fit and apply them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
from shoez.types import (
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

__all__ = [
    "DateFeaturesFit",
    "fit_date_features",
    "apply_date_features",
    "StepDateFeatures",
]


def _year(e: nw.Expr) -> nw.Expr:
    return e.dt.year()


def _month(e: nw.Expr) -> nw.Expr:
    return e.dt.month()


def _day(e: nw.Expr) -> nw.Expr:
    return e.dt.day()


def _dow(e: nw.Expr) -> nw.Expr:
    return e.dt.weekday()


def _doy(e: nw.Expr) -> nw.Expr:
    return e.dt.ordinal_day()


#: Feature name -> function computing that calendar part from a Date expression.
#: Every result is cast to ``Int32`` in `apply_date_features()` (unless the
#: feature was requested as ``categorical``) so the output dtype is identical
#: across backends: polars picks the narrowest integer width per part
#: (``Int8`` for ``day``, ``Int16`` for ``doy``, ...) while pandas' narwhals
#: backend always reports ``Int32``.
_DATE_FEATURE_EXPRS: dict[str, "Callable[[nw.Expr], nw.Expr]"] = {
    "year": _year,
    "month": _month,
    "day": _day,
    "dow": _dow,
    "doy": _doy,
}

_DEFAULT_DATE_FEATURES: tuple[str, ...] = ("year", "month", "day", "dow", "doy")

#: Ordered level labels for the two features that name a natural, fixed
#: cycle. Every other feature (``year``, ``day``, ``doy``) has no such
#: vocabulary -- there is no name for "day 214 of the year" -- so ``month``
#: and ``dow`` are the only features `categorical` accepts.
_MONTH_LABELS: list[str] = [
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
_DOW_LABELS: list[str] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]
_CATEGORICAL_LABELS: dict[str, list[str]] = {"month": _MONTH_LABELS, "dow": _DOW_LABELS}
#: Order matches ``_DEFAULT_DATE_FEATURES``' order (``month`` before ``dow``),
#: so normalizing `categorical` against this tuple keeps it consistent with
#: `features`' own canonical order.
_CATEGORICAL_ELIGIBLE: tuple[str, ...] = ("month", "dow")


def _validate_date_features(features: tuple[str, ...]) -> None:
    invalid = [f for f in features if f not in _DATE_FEATURE_EXPRS]
    if invalid:
        raise ValueError(
            f"'features' must be a subset of {sorted(_DATE_FEATURE_EXPRS)!r}, "
            f"got {invalid!r}"
        )


def _normalize_date_features(features: tuple[str, ...]) -> tuple[str, ...]:
    """Dedup ``features`` and order it canonically (`_DEFAULT_DATE_FEATURES`' order).

    Two constructor calls differing only in the order or duplication of the
    same requested parts should behave identically -- same output columns,
    same ``id`` collision behavior, same fitted state -- rather than produce
    columns in whatever order the caller happened to type them.
    """
    requested = set(features)
    return tuple(f for f in _DEFAULT_DATE_FEATURES if f in requested)


def _validate_categorical(
    categorical: tuple[str, ...], features: tuple[str, ...]
) -> None:
    invalid = [f for f in categorical if f not in _CATEGORICAL_ELIGIBLE]
    if invalid:
        raise ValueError(
            f"'categorical' must be a subset of {sorted(_CATEGORICAL_ELIGIBLE)!r}, "
            f"got {invalid!r}"
        )
    not_requested = [f for f in categorical if f not in features]
    if not_requested:
        raise ValueError(
            f"'categorical' entries must also be requested in 'features'; "
            f"not requested: {not_requested!r}"
        )


def _normalize_categorical(categorical: tuple[str, ...]) -> tuple[str, ...]:
    """Dedup ``categorical`` and order it canonically (`_CATEGORICAL_ELIGIBLE`' order)."""
    requested = set(categorical)
    return tuple(f for f in _CATEGORICAL_ELIGIBLE if f in requested)


def _feature_expr(col_name: str, feature: str, *, categorical: bool) -> nw.Expr:
    """Build the expression computing ``feature`` from ``col_name``, aliased.

    Plain features are cast to ``Int32`` for backend parity (see
    `_DATE_FEATURE_EXPRS`). A ``categorical`` feature is instead mapped to its
    ordered level labels (`_CATEGORICAL_LABELS`) and cast to a narwhals
    ``Enum`` of those labels, so ``month``/``dow`` come back with readable
    level names (``"January"``, ``"Monday"``, ...) instead of raw numbers.
    """
    name = f"{col_name}_{feature}"
    raw = _DATE_FEATURE_EXPRS[feature](nw.col(col_name))
    if categorical:
        labels = _CATEGORICAL_LABELS[feature]
        mapping = {i + 1: label for i, label in enumerate(labels)}
        return raw.replace_strict(mapping, return_dtype=nw.Enum(labels)).alias(name)
    return raw.cast(nw.Int32).alias(name)


def _output_dtype_name(feature: str, *, categorical: tuple[str, ...]) -> str:
    """The dtype string `_feature_expr()` produces for ``feature``, for `ColSchema`."""
    if feature in categorical:
        return str(nw.Enum(_CATEGORICAL_LABELS[feature]))
    return "Int32"


@dataclass(frozen=True)
class DateFeaturesFit:
    """The fitted result returned by `fit_date_features()`.

    Attributes
    ----------
    columns
        Selected column names, in fit-time order.
    features
        Calendar parts to compute for each column, in requested order.
    categorical
        Subset of ``features`` to encode as an ordered ``Enum`` of level names
        instead of a plain integer.
    """

    columns: list[str]
    features: tuple[str, ...]
    categorical: tuple[str, ...] = ()


def fit_date_features(
    selected: Frame, features: tuple[str, ...], categorical: tuple[str, ...] = ()
) -> DateFeaturesFit:
    """Record which columns and calendar parts `apply_date_features()` will compute.

    There is no per-column statistic to estimate: each calendar part is an
    elementwise function of the column's value, so this just records the
    column order and the requested feature names.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the Date columns to expand. Native
        frames are wrapped via `shoez._frame.from_native()` at the recipe
        boundary.
    features
        Calendar parts to compute; every entry must be a key of the supported
        vocabulary (validated by the caller, `StepDateFeatures.__init__`).
    categorical
        Subset of ``features`` to encode as an ordered ``Enum`` of level names
        instead of a plain integer (validated by the caller).

    Returns
    -------
    DateFeaturesFit
        The column order and feature list `apply_date_features()` needs.
    """
    return DateFeaturesFit(
        columns=list(selected.columns), features=features, categorical=categorical
    )


def apply_date_features(
    data: Frame,
    fit_result: DateFeaturesFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Replace each selected Date column with its requested calendar-part columns.

    For every column in ``fit_result.columns``, adds one column per entry of
    ``fit_result.features``, named ``{column}_{feature}`` and placed in the
    column's original slot, replacing it (or following it, kept, under
    ``keep_original_cols``). A feature named in ``fit_result.categorical`` is
    encoded as an ordered ``Enum`` of level names instead of a plain integer.
    Every other column of ``data`` passes through unchanged.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_date_features()`.
    keep_original_cols
        When ``True``, each expanded source column is left in place alongside
        its calendar-part columns instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new column per selected column per requested
        feature. Backend (pandas/polars) is preserved end-to-end.
    """
    final_order: list[str] = []
    to_drop: list[str] = []
    new_cols: list[nw.Expr] = []
    selected_cols = set(fit_result.columns)
    categorical = set(fit_result.categorical)
    for col_name in data.columns:
        if col_name not in selected_cols:
            final_order.append(col_name)
            continue
        if keep_original_cols:
            final_order.append(col_name)
        else:
            to_drop.append(col_name)
        for feature in fit_result.features:
            final_order.append(f"{col_name}_{feature}")
            new_cols.append(
                _feature_expr(col_name, feature, categorical=feature in categorical)
            )

    if new_cols:
        data = data.with_columns(*new_cols)
    if to_drop:
        data = data.drop(to_drop)
    return data.select(final_order)


class StepDateFeatures(Step):
    """Expand Date columns into calendar-part feature columns.

    Each selected Date column is replaced by one integer column per requested
    entry of ``features`` (``year``, ``month``, ``day``, ``dow``, ``doy``),
    named ``{column}_{feature}``, or kept alongside them when
    ``keep_original_cols`` is ``True``. Nothing is estimated from the training
    data: every calendar part is an elementwise function of the column's
    value, so the transform is identical on training and new data.

    ``dow`` is the ISO day of week (``1`` = Monday, ``7`` = Sunday); ``doy`` is
    the ordinal day of year (``1``-``366``). Produced columns are ``Int32``
    regardless of backend, unless named in ``categorical``.

    A Datetime column (carrying a time-of-day component) is not accepted here;
    use `StepDatetimeFeatures` for those.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to expand. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.date_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.by_dtype(narwhals.Date)``). Selectors are
        resolved against the frame's schema when the recipe is fitted, so the
        same selector works on any backend. Only Date columns are meaningful
        here.
    features
        Calendar parts to compute, from ``"year"``, ``"month"``, ``"day"``,
        ``"dow"`` (day of week), and ``"doy"`` (day of year). Default
        ``("year", "month", "day", "dow", "doy")``. Deduplicated and reordered
        to that same canonical order regardless of the order passed in, so the
        output columns are always in a fixed order.
    categorical
        Subset of ``features``, restricted to ``"month"`` and ``"dow"`` (the
        two parts with a natural, fixed vocabulary of level names), to encode
        as an ordered categorical column of level names (``"January"``,
        ``"Monday"``, ...) instead of a plain integer. Default ``()``: no
        feature is categorical. Each entry must also be present in
        ``features``.
    keep_original_cols
        If ``True``, the source columns are kept alongside their calendar-part
        columns instead of being dropped. Default ``False``.
    role
        Role assigned to the new calendar-part columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"date_features"``; a second step
        of this type in the same recipe becomes ``"date_features_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced by its calendar-part columns:

    ```{python}
    #| echo: false
    #| output: false
    import polars as pl
    from datetime import date
    from shoez import Recipe, StepDateFeatures
    from shoez import Selectors as s
    ```

    ```{python}
    df = pl.DataFrame({"d": [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDateFeatures(s.date_features()))
    recipe
    ```

    There is nothing to estimate, so `fit()` only resolves the selection:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` adds one column per requested calendar part and drops the source:

    ```{python}
    fitted.transform(df)
    ```

    A narrower ``features`` produces fewer columns:

    ```{python}
    recipe = Recipe().add(StepDateFeatures(s.date_features(), features=["year", "month"]))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside the calendar parts:

    ```{python}
    recipe = Recipe().add(
        StepDateFeatures(s.date_features(), keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``categorical`` names level labels instead of raw numbers for ``month`` and ``dow``:

    ```{python}
    recipe = Recipe().add(
        StepDateFeatures(s.date_features(), categorical=["month", "dow"])
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: Selector,
        *,
        features: list[str] | tuple[str, ...] = _DEFAULT_DATE_FEATURES,
        categorical: list[str] | tuple[str, ...] = (),
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        features = tuple(features)
        _validate_date_features(features)
        features = _normalize_date_features(features)
        categorical = tuple(categorical)
        _validate_categorical(categorical, features)
        self.columns = columns
        self.features = features
        self.categorical = _normalize_categorical(categorical)
        super().__init__(
            id,
            "date_features",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )
        # Populated by ``fit()``.
        self._estimates: DateFeaturesFit | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record which columns and calendar parts to compute."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_date_features(selected, self.features, self.categorical)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        out_cols = [
            f"{col}_{feature}" for col in selected.columns for feature in self.features
        ]
        out_dtypes = {
            f"{col}_{feature}": _output_dtype_name(
                feature, categorical=self.categorical
            )
            for col in selected.columns
            for feature in self.features
        }
        self._schema_out = ColSchema(columns=out_cols, dtypes=out_dtypes)
        self._output_parents = {
            f"{col}_{feature}": [col]
            for col in selected.columns
            for feature in self.features
        }
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Add each requested calendar-part column. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepDateFeatures.transform() called before fit()")
        return apply_date_features(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDateFeatures.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the calendar-part columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepDateFeatures.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each calendar-part column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepDateFeatures.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Calendar-part expansion is only meaningful on Date columns."""
        return {nw.Date}

    def update_roles(self, roles: Roles) -> None:
        if self._output_parents is None:
            return
        if self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)
        if self.keep_original_cols:
            return
        expanded = {
            parent for parents in self._output_parents.values() for parent in parents
        }
        for col in expanded:
            for role in list(roles.of(col)):
                roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """Calendar-part columns are dense by construction."""
        return False
