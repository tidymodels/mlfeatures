"""Replace missing categorical levels with an explicit ``"unknown"`` level.

The *fit* half lives in `fit_unknown()` and the *apply* half in
`apply_unknown()`, both plain functions so the data-ops core can be
exercised on its own. `StepUnknown` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

Each selected categorical column has its null values recoded to a single
explicit level so that "missing" becomes a value a model can learn from rather
than a gap. Ports R ``recipes``' ``step_unknown``.

Deviation from R (for now): recoded columns are returned as ``String`` rather
than as a categorical with an added level. Preserving the categorical dtype and
its level set is deferred until the schema subsystem lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures._level_recode import warn_level_collision
from mlfeatures.step import Step
from mlfeatures.types import (
    _CATEGORICAL_DTYPES,  # pyright: ignore[reportPrivateUsage]
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_unknown", "apply_unknown", "StepUnknown"]

#: Default level used to replace missing categorical values.
DEFAULT_UNKNOWN_LEVEL = "unknown"


def fit_unknown(
    selected: Frame,
    new_level: str = DEFAULT_UNKNOWN_LEVEL,
) -> list[str]:
    """Determine which columns to recode and check for level collisions.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all categorical-like. Passed in
        by the step's ``fit()`` method after selector resolution.
    new_level:
        The level that null values will be recoded to. Default ``"unknown"``.

    Returns
    -------
    list[str] — the column names to recode, in input order. Every selected
    column is returned (not only those with nulls in the training data), so that
    nulls appearing at transform time are still recoded — guaranteeing
    train/serve parity.

    A `UserWarning` is emitted if ``new_level`` already appears as a value
    in any column, since recoding would then merge genuine samples into the
    "unknown" bucket.
    """
    collisions = [
        col
        for col in selected.columns
        if bool((selected[col].cast(nw.String) == new_level).any())
    ]
    warn_level_collision(new_level, collisions, merges="recoded nulls")
    return list(selected.columns)


def apply_unknown(
    selected: Frame,
    columns: list[str],
    new_level: str = DEFAULT_UNKNOWN_LEVEL,
) -> Frame:
    """Recode null values in ``columns`` to ``new_level``.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to recode.
    columns:
        The column names to recode — the columns returned by `fit_unknown()`.
        Any column of ``selected`` not in this list is passed through untouched.
    new_level:
        The level that null values are recoded to. Default ``"unknown"``.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``columns`` cast to
    ``String`` and its nulls filled with ``new_level``. Column names, order, and
    structure are unchanged.
    """
    # Recode nulls with an explicit ``is_null`` mask rather than casting first
    # and relying on ``fill_null``: older polars (resolved on Python 3.10) casts
    # null to the literal string ``"None"`` instead of preserving it as null,
    # which would leave nothing for ``fill_null`` to replace.
    exprs = [
        nw.when(~nw.col(col).is_null())
        .then(nw.col(col).cast(nw.String))
        .otherwise(nw.lit(new_level))
        .alias(col)
        for col in columns
    ]
    return selected.with_columns(*exprs)


class StepUnknown(Step):
    """Recode missing categorical levels to an explicit ``"unknown"`` level.

    Each selected categorical column has its nulls replaced in place by
    ``new_level``.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to recode. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.nominal_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.string()``, ``polars.selectors.string()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only categorical
        columns are meaningful here.
    new_level
        The level that replaces nulls. Default ``"unknown"``. A `UserWarning` is
        emitted if a selected column already contains this level, since the
        recoded rows then become indistinguishable from real ones.
    id
        Stable step identifier. Defaults to ``"unknown"``; a second step of
        this type in the same recipe becomes ``"unknown_2"``, and passing an
        ``id`` already used in the recipe is an error.

    Examples
    --------
    Nulls in the selected column are recoded in place to an explicit
    ``"unknown"`` level:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepUnknown
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {"city": pd.Series(["NYC", "LA", "NYC", None], dtype="category")}
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepUnknown(s.nominal_features()))
    recipe
    ```

    `fit()` records the level to add to each column:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` recodes the nulls in place:

    ```{python}
    fitted.transform(df)
    ```

    ``new_level`` names the added level:

    ```{python}
    recipe = Recipe().add(StepUnknown(s.nominal_features(), new_level="missing"))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        *,
        new_level: str = DEFAULT_UNKNOWN_LEVEL,
        id: str | None = None,
    ):
        self.columns = columns
        self.new_level = new_level
        super().__init__(id, "unknown")

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the columns to recode and check for level collisions."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        # Called for its collision check / warning; every selected column is
        # recoded, so the resolved selection is recorded in ``_schema_in``.
        fit_unknown(selected, self.new_level)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Recode nulls in the fitted columns to ``new_level``. Never re-estimates."""
        if self._schema_in is None:
            raise RuntimeError("StepUnknown.transform() called before fit()")
        return apply_unknown(new_data, self._schema_in.columns, self.new_level)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepUnknown.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, all recoded to ``String``."""
        if self._schema_in is None:
            raise RuntimeError("StepUnknown.schema_out() called before fit()")
        # Every column in the resolved selection is recoded to String.
        dtypes = dict.fromkeys(self._schema_in.columns, "String")
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Recoding missing levels is only meaningful on categorical columns."""
        return set(_CATEGORICAL_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_unknown`` never produces sparse output."""
        return False
