"""Holiday indicator features from a Date or Datetime column: fit + apply.

Whether a given date falls on a named holiday is an elementwise property of
the value, not a training-time statistic, so the *fit* half
(`fit_holiday()`) has nothing to estimate either: it only resolves the
selected columns and checks the requested holiday names against a real
calendar built from the training data's own year span, so a typo in
``holidays`` surfaces at fit time rather than silently producing an
all-zero column. The *apply* half (`apply_holiday()`) rebuilds the calendar
from whatever data it is given (training or new) so a date outside the
training year range is still classified correctly. `StepHoliday` wraps both
into the `Step` lifecycle so a `Recipe` can fit and apply them.

Holiday data comes from the `holidays` PyPI package's per-country calendars
(see ``shoez._dependencies.import_optional``). Unlike R recipes' ``timeDate``
backend, which exposes one function per holiday across all countries at
once, `holidays` groups holidays under a single country (and optional
subdivision), so ``holidays`` names here are only meaningful relative to the
step's own ``country``/``subdiv``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez._dependencies import import_optional
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
    from typing_extensions import Self

__all__ = [
    "HolidayFit",
    "fit_holiday",
    "apply_holiday",
    "StepHoliday",
]

_DEFAULT_HOLIDAYS: tuple[str, ...] = ("New Year's Day", "Labor Day", "Christmas Day")

_SLUG_INVALID = re.compile(r"[^a-z0-9]+")


def _slug(name: str) -> str:
    """A column-name-safe slug for a holiday name (``"New Year's Day"`` -> ``"new_years_day"``)."""
    return _SLUG_INVALID.sub("_", name.lower().replace("'", "")).strip("_")


def _year_span(data: Frame, column: str) -> "tuple[int, int] | None":
    """The ``(min_year, max_year)`` spanned by ``column``'s non-null values, or ``None`` if none."""
    series = data[column]
    if series.len() - series.null_count() == 0:
        return None
    years = series.dt.year()
    return int(years.min()), int(years.max())


def _date_expr(column: str, dtype: nw.dtypes.DType) -> nw.Expr:
    """A ``Date``-valued expression for ``column``, converting a Datetime down to its date part."""
    col = nw.col(column)
    return col.dt.date() if isinstance(dtype, nw.Datetime) else col


@dataclass(frozen=True)
class HolidayFit:
    """The fitted result returned by `fit_holiday()`.

    Attributes
    ----------
    columns
        Selected column names, in fit-time order.
    holidays
        Holiday names to flag, in requested order.
    country
        `holidays` package country code the holiday names are resolved against.
    subdiv
        Subdivision code within ``country``, or ``None``.
    dtypes
        Each selected column's dtype at fit time, keyed by name.
    """

    columns: list[str]
    holidays: tuple[str, ...]
    country: str
    subdiv: "str | None"
    dtypes: dict[str, nw.dtypes.DType]


def fit_holiday(
    selected: Frame,
    holidays: tuple[str, ...],
    country: str,
    subdiv: "str | None" = None,
) -> HolidayFit:
    """Resolve the selected columns and validate ``holidays`` against a real calendar.

    Builds a `holidays` package calendar for ``country``/``subdiv`` over the
    year span actually present in ``selected``, then checks every requested
    holiday name appears in it; an unrecognized name raises `ValueError`
    rather than silently producing an all-zero indicator column later. If
    every selected column is entirely null, there is no year span to build a
    calendar from, so this validation is skipped.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the Date/Datetime columns to flag
        holidays on. Native frames are wrapped via `shoez._frame.from_native()`
        at the recipe boundary.
    holidays
        Holiday names to flag; each must be a name the `holidays` package
        reports for ``country``/``subdiv`` in at least one of the years
        present in ``selected`` (validated by the caller,
        `StepHoliday.__init__`, only checks that this argument is a tuple).
    country
        `holidays` package country code, e.g. ``"US"``.
    subdiv
        Subdivision code within ``country``, e.g. a US state code, or ``None``.

    Returns
    -------
    HolidayFit
        The column order, holiday list, and calendar location
        `apply_holiday()` needs.
    """
    holidays_module = import_optional("holidays", purpose="by StepHoliday")
    spans = [_year_span(selected, col) for col in selected.columns]
    lo_years = [span[0] for span in spans if span is not None]
    hi_years = [span[1] for span in spans if span is not None]
    if lo_years:
        years = list(range(min(lo_years), max(hi_years) + 1))
        calendar = holidays_module.country_holidays(country, subdiv=subdiv, years=years)
        known = set(calendar.values())
        unknown = [name for name in holidays if name not in known]
        if unknown:
            raise ValueError(
                f"'holidays' must be holiday names in the {country!r} calendar "
                f"for {years[0]}-{years[-1]}; got unknown name(s) {unknown!r}"
            )
    return HolidayFit(
        columns=list(selected.columns),
        holidays=holidays,
        country=country,
        subdiv=subdiv,
        dtypes=_frame.dtypes(selected),
    )


def apply_holiday(
    data: Frame,
    fit_result: HolidayFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Replace each selected column with one holiday-indicator column per holiday name.

    For every column in ``fit_result.columns``, adds one ``Int32`` column per
    entry of ``fit_result.holidays``, named ``{column}_{slug}`` (a
    column-safe slug of the holiday name), placed in the column's original
    slot, replacing it (or following it, kept, under ``keep_original_cols``).
    A holiday's calendar is rebuilt from ``data``'s own year span on every
    call, so new data outside the year span seen at fit time is still
    classified correctly. Every other column of ``data`` passes through
    unchanged.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_holiday()`.
    keep_original_cols
        When ``True``, each source column is left in place alongside its
        holiday-indicator columns instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new ``Int32`` column per selected column per
        requested holiday. Backend (pandas/polars) is preserved end-to-end.
    """
    holidays_module = import_optional("holidays", purpose="by StepHoliday")
    final_order: list[str] = []
    to_drop: list[str] = []
    new_cols: list[nw.Expr] = []
    selected_cols = set(fit_result.columns)
    for col_name in data.columns:
        if col_name not in selected_cols:
            final_order.append(col_name)
            continue
        if keep_original_cols:
            final_order.append(col_name)
        else:
            to_drop.append(col_name)

        span = _year_span(data, col_name)
        years = list(range(span[0], span[1] + 1)) if span is not None else []
        calendar = holidays_module.country_holidays(
            fit_result.country, subdiv=fit_result.subdiv, years=years
        )
        date_expr = _date_expr(col_name, fit_result.dtypes[col_name])
        for name in fit_result.holidays:
            matching: list[Any] = [d for d, n in calendar.items() if n == name]
            final_order.append(f"{col_name}_{_slug(name)}")
            new_cols.append(
                date_expr.is_in(matching)
                .cast(nw.Int32)
                .alias(f"{col_name}_{_slug(name)}")
            )

    if new_cols:
        data = data.with_columns(*new_cols)
    if to_drop:
        data = data.drop(to_drop)
    return data.select(final_order)


class StepHoliday(Step):
    """Add binary indicator columns flagging named holidays.

    Each selected Date/Datetime column gets one companion ``Int32`` column
    per requested holiday name, named ``{column}_{holiday}`` (the holiday
    name lowercased with non-alphanumeric characters collapsed to
    underscores), that is ``1`` where the value falls on that holiday and
    ``0`` otherwise, or kept alongside the source when ``keep_original_cols``
    is ``True``. A Datetime value is compared by its date part; time of day
    never affects the result.

    Holidays are looked up in the `holidays` PyPI package's calendar for
    ``country`` (and, if given, ``subdiv``), rebuilt from the actual year
    span present in the data being transformed, so dates outside the
    training year range are still classified correctly. A holiday name is
    only meaningful relative to the ``country``/``subdiv`` it was requested
    with; some public holidays shift onto an adjacent weekday and are
    reported under a separate ``"... (observed)"`` name, which only matches
    if requested explicitly.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to flag. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.date_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.by_dtype(narwhals.Date)``). Selectors are
        resolved against the frame's schema when the recipe is fitted, so the
        same selector works on any backend. Only Date and Datetime columns
        are meaningful here.
    holidays
        Holiday names to flag, matched against the `holidays` package's
        calendar for ``country``/``subdiv``. Default
        ``("New Year's Day", "Labor Day", "Christmas Day")``.
    country
        `holidays` package country code the holiday names are resolved
        against. Default ``"US"``.
    subdiv
        Subdivision code within ``country`` (e.g. a US state or Canadian
        province code), for holidays observed only in part of a country.
        Default ``None``.
    keep_original_cols
        If ``True``, the source columns are kept alongside their
        holiday-indicator columns instead of being dropped. Default
        ``False``.
    role
        Role assigned to the new indicator columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"holiday"``; a second step of
        this type in the same recipe becomes ``"holiday_2"``, and passing an
        ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column gets one indicator column per requested holiday:

    ```{python}
    #| echo: false
    #| output: false
    import polars as pl
    from datetime import date
    from shoez import Recipe, StepHoliday
    from shoez import Selectors as s
    ```

    ```{python}
    df = pl.DataFrame(
        {"d": [date(2023, 12, 24), date(2023, 12, 25), date(2024, 1, 1)]}
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepHoliday(s.date_features(), holidays=["Christmas Day"]))
    recipe
    ```

    `fit()` validates the holiday names against a real calendar built from the training data's years:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` adds one indicator column per requested holiday and drops the source:

    ```{python}
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside the indicators:

    ```{python}
    recipe = Recipe().add(
        StepHoliday(
            s.date_features(), holidays=["Christmas Day"], keep_original_cols=True
        )
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``country`` resolves the same holiday names against a different calendar:

    ```{python}
    recipe = Recipe().add(
        StepHoliday(s.date_features(), holidays=["Christmas Day"], country="CA")
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
        holidays: "list[str] | tuple[str, ...]" = _DEFAULT_HOLIDAYS,
        country: str = "US",
        subdiv: "str | None" = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.holidays = tuple(holidays)
        self.country = country
        self.subdiv = subdiv
        super().__init__(
            id,
            "holiday",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )
        # Populated by ``fit()``.
        self._estimates: HolidayFit | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the selection and validate ``holidays`` against a real calendar."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_holiday(
            selected, self.holidays, self.country, self.subdiv
        )
        self._schema_in = ColSchema(
            columns=selected.columns,
            dtypes={name: str(dt) for name, dt in _frame.dtypes(selected).items()},
        )
        out_cols = [
            f"{col}_{_slug(name)}" for col in selected.columns for name in self.holidays
        ]
        out_dtypes = {name: "Int32" for name in out_cols}
        self._schema_out = ColSchema(columns=out_cols, dtypes=out_dtypes)
        self._output_parents = {
            f"{col}_{_slug(name)}": [col]
            for col in selected.columns
            for name in self.holidays
        }
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Add each requested holiday-indicator column. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepHoliday.transform() called before fit()")
        return apply_holiday(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepHoliday.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the holiday-indicator columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepHoliday.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each holiday-indicator column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepHoliday.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Holiday flagging is only meaningful on Date/Datetime columns."""
        return {nw.Date, nw.Datetime}

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

    def required_pkgs(self) -> list[str]:
        """`StepHoliday` needs the `holidays` package to look up calendars."""
        return ["holidays"]

    @property
    def produces_sparse(self) -> bool:
        """Holiday-indicator columns are dense by construction."""
        return False
