"""Impute missing values with the column mode (most frequent value).

The *fit* half lives in `fit_impute_mode()` and the *apply* half in
`apply_impute_mode()`, both plain functions so the data-ops core can be
exercised on its own. `StepImputeMode` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

Each selected categorical column has its null values filled with that column's
most frequent level. Ports R ``recipes``' ``step_impute_mode``, which likewise
targets nominal variables.

Mode imputation is for **categorical** columns only — the most frequent observed
level. (The mean/median steps cover numeric columns; the mode of a continuous
numeric column is not meaningful.) It **preserves** the column's dtype, since the
fill value is an observed level of the column.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
from shoez.types import AcceptedDtypes, ColSchema, Frame, Mode, SchemaInfo, Selector

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_impute_mode", "apply_impute_mode", "StepImputeMode"]

#: Dtypes ``step_impute_mode`` is valid on — categorical-like columns.
_CATEGORICAL_DTYPES: set[type[nw.dtypes.DType]] = {nw.Categorical, nw.Enum, nw.String}


def fit_impute_mode(selected: Frame) -> dict[str, Any]:
    """Estimate the per-column mode used to impute missing values.

    Parameters
    ----------
    selected:
        A narwhals DataFrame of categorical columns. Passed in by the step's
        ``fit()`` method after selector resolution; dtype screening is the
        ``accepts_dtypes`` layer's job, so this function does not re-validate it.

    Returns
    -------
    ``dict`` mapping every input column name to its mode. Empty when
    ``selected`` has no columns. When a column has several equally-frequent
    values, the smallest (by sort order) is chosen so the result is
    deterministic.

    A column whose values are *all* null has no mode and so cannot be imputed.
    It is **not** dropped: its value in the mapping is ``None``,
    `apply_impute_mode()` leaves it unchanged, and a `UserWarning`
    is emitted naming such columns.
    """
    modes: dict[str, Any] = {}
    all_null: list[str] = []

    for col in selected.columns:
        # Drop nulls first so a null is never itself selected as the mode, then
        # break ties deterministically by taking the smallest value.
        candidates = [
            v
            for v in selected[col].drop_nulls().mode().sort().to_list()
            if v is not None
        ]
        if candidates:
            modes[col] = candidates[0]
        else:
            modes[col] = None
            all_null.append(col)

    if all_null:
        warnings.warn(
            f"{len(all_null)} column(s) are entirely null and have no mode to "
            f"impute with; left unchanged: {all_null!r}",
            stacklevel=2,
        )

    return modes


def apply_impute_mode(
    selected: Frame,
    modes: dict[str, Any],
) -> Frame:
    """Fill nulls with the stored per-column mode.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to impute.
    modes:
        The ``{column: mode}`` mapping returned by `fit_impute_mode()`. Any
        column of ``selected`` not present as a key is passed through untouched.
        A ``None`` mode marks a column that had no mode (entirely null at fit
        time); such columns are left unchanged.

    Returns
    -------
    narwhals.DataFrame — the input frame with nulls in each mapped column filled
    by its mode (columns with a ``None`` mode are left as-is). Column names,
    order, dtypes, and structure are unchanged.
    """
    # Skip columns with a None mode (entirely null at fit time): leave as-is.
    exprs = [
        nw.col(col).fill_null(mode).alias(col)
        for col, mode in modes.items()
        if mode is not None
    ]
    return selected.with_columns(*exprs)


class StepImputeMode(Step):
    """Impute missing values with the column mode (most frequent value).

    Each selected column has its nulls filled in place with that column's
    mode.

    Mode imputation is for categorical columns (it accepts
    ``Categorical``/``Enum``/``String``, not numeric dtypes) and is
    dtype-preserving — the fill value is an observed level of the column.

    A column that is entirely null at fit time has no mode; it is left unchanged
    (not dropped) with a warning, and passes through ``transform`` untouched.

    Parameters
    ----------
    columns
        Columns to impute. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.nominal_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.string()``, ``polars.selectors.string()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only categorical
        columns are meaningful here.
    id
        Stable step identifier. Default ``"impute_mode"``.

    Examples
    --------
    The null is filled in place with the column mode (the most frequent
    observed level):

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepImputeMode
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {"city": pd.Series(["NYC", "LA", "NYC", None], dtype="category")}
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepImputeMode(s.nominal_features()))
    recipe
    ```

    `fit()` finds the most frequent observed level:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` fills the nulls in place:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        id: str | None = None,
    ):
        self.columns = columns
        self.id = id if id is not None else "impute_mode"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: {column: mode} over the imputable columns.
        self._estimates: dict[str, Any] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate the per-column mode over the selected columns."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_impute_mode(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Fill nulls with the stored modes. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepImputeMode.transform() called before fit()")
        return apply_impute_mode(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMode.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns — unchanged: mode imputation preserves dtype."""
        if self._schema_in is None:
            raise RuntimeError("StepImputeMode.schema_out() called before fit()")
        return ColSchema(
            columns=self._schema_in.columns, dtypes=dict(self._schema_in.dtypes)
        )

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Mode imputation is only meaningful on categorical columns."""
        return set(_CATEGORICAL_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_impute_mode`` never produces sparse output."""
        return False
