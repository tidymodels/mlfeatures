"""Calendar-part expansion for Datetime columns: fit + apply.

`step_datetime_features` is the Datetime-only counterpart of
`step_date_features` (`shoez/step_date_features.py`). They are two separate
steps because a date has no time of day: ``hour``, ``minute``, and ``second``
make no sense on a plain calendar date, so a single step trying to cover both
would need to either fake a time-of-day or special-case Date columns
internally. Splitting mirrors Python's own ``datetime.date`` /
``datetime.datetime`` split (``datetime`` does not inherit from ``date``).

Time zones are read as the backend already resolves them: this step does not
accept a target zone to convert into, so a column's stored zone (including
none) governs the calendar parts it produces. Zone conversion is left to a
caller who wants it, via ``narwhals.Expr.dt.convert_time_zone()`` in a
`StepMutate`, rather than added as a constructor argument here.

The *fit* half (`fit_datetime_features()`) only needs the selected columns'
names and the requested feature list; there is nothing to estimate from the
data. The *apply* half (`apply_datetime_features()`) computes each requested
calendar part. `StepDatetimeFeatures` wraps both into the `Step` lifecycle so
a `Recipe` can fit and apply them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez._calendar_features import (
    apply_calendar_features,
    day_expr,
    dow_expr,
    doy_expr,
    month_expr,
    normalize_categorical,
    normalize_features,
    output_dtype_name,
    validate_categorical,
    validate_features,
    year_expr,
)
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
    "DatetimeFeaturesFit",
    "fit_datetime_features",
    "apply_datetime_features",
    "StepDatetimeFeatures",
]


def _hour(e: nw.Expr) -> nw.Expr:
    return e.dt.hour()


def _minute(e: nw.Expr) -> nw.Expr:
    return e.dt.minute()


def _second(e: nw.Expr) -> nw.Expr:
    return e.dt.second()


#: Feature name -> function computing that calendar part from a Datetime
#: expression. Every result is cast to ``Int32`` in `apply_datetime_features()`
#: so the output dtype is identical across backends: polars picks the
#: narrowest integer width per part while pandas' narwhals backend always
#: reports ``Int32``.
_DATETIME_FEATURE_EXPRS: dict[str, "Callable[[nw.Expr], nw.Expr]"] = {
    "year": year_expr,
    "month": month_expr,
    "day": day_expr,
    "dow": dow_expr,
    "doy": doy_expr,
    "hour": _hour,
    "minute": _minute,
    "second": _second,
}

_DEFAULT_DATETIME_FEATURES: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "dow",
    "doy",
)


@dataclass(frozen=True)
class DatetimeFeaturesFit:
    """The fitted result returned by `fit_datetime_features()`.

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


def fit_datetime_features(
    selected: Frame, features: tuple[str, ...], categorical: tuple[str, ...] = ()
) -> DatetimeFeaturesFit:
    """Record which columns and calendar parts `apply_datetime_features()` will compute.

    There is no per-column statistic to estimate: each calendar part is an
    elementwise function of the column's value, so this just records the
    column order and the requested feature names.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the Datetime columns to expand.
        Native frames are wrapped via `shoez._frame.from_native()` at the
        recipe boundary.
    features
        Calendar parts to compute; every entry must be a key of the supported
        vocabulary (validated by the caller, `StepDatetimeFeatures.__init__`).
    categorical
        Subset of ``features`` to encode as an ordered ``Enum`` of level names
        instead of a plain integer (validated by the caller).

    Returns
    -------
    DatetimeFeaturesFit
        The column order and feature list `apply_datetime_features()` needs.
    """
    return DatetimeFeaturesFit(
        columns=list(selected.columns), features=features, categorical=categorical
    )


def apply_datetime_features(
    data: Frame,
    fit_result: DatetimeFeaturesFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Replace each selected Datetime column with its requested calendar-part columns.

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
        Output of `fit_datetime_features()`.
    keep_original_cols
        When ``True``, each expanded source column is left in place alongside
        its calendar-part columns instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new column per selected column per requested
        feature. Backend (pandas/polars) is preserved end-to-end.
    """
    return apply_calendar_features(
        data,
        fit_result.columns,
        fit_result.features,
        fit_result.categorical,
        _DATETIME_FEATURE_EXPRS,
        keep_original_cols=keep_original_cols,
    )


class StepDatetimeFeatures(Step):
    """Expand Datetime columns into calendar-part feature columns.

    Each selected Datetime column is replaced by one integer column per
    requested entry of ``features`` (``year``, ``month``, ``day``, ``hour``,
    ``minute``, ``second``, ``dow``, ``doy``), named ``{column}_{feature}``,
    or kept alongside them when ``keep_original_cols`` is ``True``. Nothing is
    estimated from the training data: every calendar part is an elementwise
    function of the column's value, so the transform is identical on training
    and new data.

    ``dow`` is the ISO day of week (``1`` = Monday, ``7`` = Sunday); ``doy`` is
    the ordinal day of year (``1``-``366``). Produced columns are ``Int32``
    regardless of backend, unless named in ``categorical``.

    Calendar parts are read against whatever time zone the column already
    carries (including none); this step does not convert to a different zone.
    A caller who needs a specific zone should convert it first, e.g. with
    ``narwhals.Expr.dt.convert_time_zone()`` inside a `StepMutate`.

    A Date column (with no time-of-day component) is not accepted here; use
    `StepDateFeatures` for those.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to expand. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.datetime_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.by_dtype(narwhals.Datetime)``). Selectors are
        resolved against the frame's schema when the recipe is fitted, so the
        same selector works on any backend. Only Datetime columns are
        meaningful here.
    features
        Calendar parts to compute, from ``"year"``, ``"month"``, ``"day"``,
        ``"hour"``, ``"minute"``, ``"second"``, ``"dow"`` (day of week), and
        ``"doy"`` (day of year). Default
        ``("year", "month", "day", "hour", "minute", "second", "dow", "doy")``.
        Deduplicated and reordered to that same canonical order regardless of
        the order passed in, so the output columns are always in a fixed
        order.
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
        Stable step identifier. Defaults to ``"datetime_features"``; a second
        step of this type in the same recipe becomes
        ``"datetime_features_2"``, and passing an ``id`` already used in the
        recipe is an error.

    Examples
    --------
    Each selected column is replaced by its calendar-part columns:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from datetime import datetime
    from shoez import Recipe, StepDatetimeFeatures
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {
            "dt": [
                datetime(2020, 1, 1, 8, 30, 0),
                datetime(2020, 6, 15, 13, 45, 30),
                datetime(2020, 12, 31, 23, 59, 59),
            ]
        }
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDatetimeFeatures(s.datetime_features()))
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
    recipe = Recipe().add(
        StepDatetimeFeatures(s.datetime_features(), features=["year", "month", "hour"])
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside the calendar parts:

    ```{python}
    recipe = Recipe().add(
        StepDatetimeFeatures(s.datetime_features(), keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``categorical`` names level labels instead of raw numbers for ``month`` and ``dow``:

    ```{python}
    recipe = Recipe().add(
        StepDatetimeFeatures(s.datetime_features(), categorical=["month", "dow"])
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
        features: list[str] | tuple[str, ...] = _DEFAULT_DATETIME_FEATURES,
        categorical: list[str] | tuple[str, ...] = (),
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        features = tuple(features)
        validate_features(features, _DATETIME_FEATURE_EXPRS)
        features = normalize_features(features, _DEFAULT_DATETIME_FEATURES)
        categorical = tuple(categorical)
        validate_categorical(categorical, features)
        self.columns = columns
        self.features = features
        self.categorical = normalize_categorical(categorical)
        super().__init__(
            id,
            "datetime_features",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )
        # Populated by ``fit()``.
        self._estimates: DatetimeFeaturesFit | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record which columns and calendar parts to compute."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_datetime_features(
            selected, self.features, self.categorical
        )
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        out_cols = [
            f"{col}_{feature}" for col in selected.columns for feature in self.features
        ]
        out_dtypes = {
            f"{col}_{feature}": output_dtype_name(feature, categorical=self.categorical)
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
            raise RuntimeError("StepDatetimeFeatures.transform() called before fit()")
        return apply_datetime_features(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDatetimeFeatures.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the calendar-part columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepDatetimeFeatures.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each calendar-part column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError(
                "StepDatetimeFeatures.output_parents() called before fit()"
            )
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Calendar-part expansion is only meaningful on Datetime columns."""
        return {nw.Datetime}

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
