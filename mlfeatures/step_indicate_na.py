"""Missing-data indicator columns: fit + apply.

The *fit* half (`fit_indicate_na()`) only needs the selected columns' names;
there is nothing to estimate from the data, since "is this value missing" is
an elementwise property of each row rather than a training-time statistic.
The *apply* half (`apply_indicate_na()`) computes, for each selected column, a
binary column that is ``1`` where the source value is missing and ``0``
otherwise. `StepIndicateNa` wraps both into the `Step` lifecycle so a `Recipe`
can fit and apply them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures.step import Step
from mlfeatures.types import (
    ANY_DTYPE,
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
    "IndicateNaFit",
    "fit_indicate_na",
    "apply_indicate_na",
    "StepIndicateNa",
]

#: Dtypes for which ``is_nan()`` is meaningful; every other dtype relies on
#: ``is_null()`` alone (``is_nan()`` raises on non-float dtypes in narwhals).
_FLOAT_DTYPES = (nw.Float32, nw.Float64)


def _missing_expr(col_name: str, dtype: nw.dtypes.DType) -> nw.Expr:
    """Build the expression flagging ``col_name`` as missing, aliased ``{col}_na``.

    A float column is missing on either a null or a NaN value; every other
    dtype has no NaN representation, so ``is_null()`` alone is checked (and
    calling ``is_nan()`` on it would raise).
    """
    col = nw.col(col_name)
    if isinstance(dtype, _FLOAT_DTYPES):
        is_missing = col.is_null() | col.is_nan()
    else:
        is_missing = col.is_null()
    return is_missing.cast(nw.Int32).alias(f"{col_name}_na")


@dataclass(frozen=True)
class IndicateNaFit:
    """The fitted result returned by `fit_indicate_na()`.

    Attributes
    ----------
    columns
        Selected column names, in fit-time order.
    dtypes
        Each selected column's dtype at fit time, keyed by name.
    """

    columns: list[str]
    dtypes: dict[str, nw.dtypes.DType]


def fit_indicate_na(selected: Frame) -> IndicateNaFit:
    """Record which columns `apply_indicate_na()` will build indicators for.

    There is no per-column statistic to estimate: missingness is an
    elementwise property of each row's value, so this just records the column
    order and dtypes.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the columns to build indicators
        for. Native data frames are wrapped via `mlfeatures._frame.from_native()` at the
        recipe boundary.

    Returns
    -------
    IndicateNaFit
        The column order and dtypes `apply_indicate_na()` needs.
    """
    dtypes = _frame.dtypes(selected)
    return IndicateNaFit(columns=list(selected.columns), dtypes=dtypes)


def apply_indicate_na(
    data: Frame,
    fit_result: IndicateNaFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Add one missing-data indicator column per selected column.

    For every column in ``fit_result.columns``, adds a column named
    ``{column}_na`` that is ``1`` where the source value is missing and ``0``
    otherwise, placed in the column's original slot, replacing it (or
    following it, kept, under ``keep_original_cols``). Every other column of
    ``data`` passes through unchanged.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_indicate_na()`.
    keep_original_cols
        When ``True``, each source column is left in place alongside its
        indicator column instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new ``Int32`` column per selected column. Backend
        (pandas/polars) is preserved end-to-end.
    """
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
        final_order.append(f"{col_name}_na")
        new_cols.append(_missing_expr(col_name, fit_result.dtypes[col_name]))

    if new_cols:
        data = data.with_columns(*new_cols)
    if to_drop:
        data = data.drop(to_drop)
    return data.select(final_order)


class StepIndicateNa(Step):
    """Add a binary indicator column flagging missing values.

    Each selected column gets a companion ``{column}_na`` column that is
    ``1`` where the source value is missing and ``0`` otherwise. Nothing is
    estimated from the training data: missingness is an elementwise property
    of each row's value, so the transform is identical on training and new
    data, and a column with no missing values at fit time still gets an
    indicator (new data may have missing values training data did not).

    A float column is flagged missing on either a null or a NaN value; every
    other dtype has no NaN representation, so only null counts.

    Unlike most steps that replace a column with several, ``keep_original_cols``
    is usually worth setting to ``True`` here: the indicator alone discards the source
    column's own value, and a downstream imputation step (`StepImputeMean`,
    `StepImputeMedian`, `StepImputeMode`, `StepUnknown`) still needs that
    column to fill in.

    See [Missing Values Indicators](https://feaz-book.com/missing-indicator) in
    *Feature Engineering A to Z* for background on this technique.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to flag. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.features()`` (see `Selectors` for
        the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the data frame's schema when the recipe is
        fitted, so the same selector works on any backend. Any dtype is
        accepted.
    keep_original_cols
        If ``True``, the source columns are kept alongside their indicator
        columns instead of being dropped. Default ``False``.
    role
        Role assigned to the new indicator columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"indicate_na"``; a second step
        of this type in the same recipe becomes ``"indicate_na_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each selected column gets a companion ``_na`` indicator:

    ```{python}
    #| echo: false
    #| output: false
    import polars as pl
    from mlfeatures import Recipe, StepIndicateNa
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pl.DataFrame({"x": [1.0, None, 3.0], "g": ["a", "b", None]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepIndicateNa(s.features(), keep_original_cols=True))
    recipe
    ```

    There is nothing to estimate, so `fit()` only resolves the selection:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` adds one indicator column per selected column, keeping the source
    alongside it so it is still available to a later imputation step:

    ```{python}
    fitted.transform(df)
    ```

    Without ``keep_original_cols``, the source column is dropped instead:

    ```{python}
    recipe = Recipe().add(StepIndicateNa(s.features()))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    #: Populated by ``fit()``: the resolved columns and their fit-time dtypes.
    _estimates: IndicateNaFit | None

    def __init__(
        self,
        columns: Selector,
        *,
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        super().__init__(
            id,
            "indicate_na",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record which columns to build missing-data indicators for."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_indicate_na(selected)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        out_cols = [f"{col}_na" for col in selected.columns]
        out_dtypes = {f"{col}_na": "Int32" for col in selected.columns}
        self._schema_out = ColSchema(columns=out_cols, dtypes=out_dtypes)
        self._output_parents = {f"{col}_na": [col] for col in selected.columns}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Add each missing-data indicator column. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepIndicateNa.transform() called before fit()")
        return apply_indicate_na(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepIndicateNa.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the indicator columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepIndicateNa.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each indicator column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepIndicateNa.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Missingness is meaningful for any dtype, not just numerics."""
        return ANY_DTYPE

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
        """Indicator columns are dense by construction."""
        return False
