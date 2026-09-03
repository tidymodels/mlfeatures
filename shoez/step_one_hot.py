"""One-hot encoding: expand discrete columns into k indicator columns.

``fit_one_hot`` identifies all unique levels per column (no reference level
dropped). Transform logic is shared with `shoez.step_dummy` via
`apply_dummy()`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
from shoez.step_dummy import (
    DISCRETE_DTYPES,
    apply_dummy,
    dummy_output_parents,
    dummy_update_roles,
    resolve_col_name,
)
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

__all__ = ["fit_one_hot", "StepOneHot"]


def fit_one_hot(selected: nw.DataFrame[Any]) -> dict[str, list[Any]]:
    """Identify all levels for one-hot (full indicator) encoding.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all discrete (string, integer,
        boolean, categorical, or enum). Passed in by `StepOneHot.fit()`
        after column selection.

    Returns
    -------
    dict[str, list[Any]]
        Maps each column name to its full sorted level list (all *k* levels
        retained; no reference level is dropped).

    Raises
    ------
    ValueError
        If the frame has zero rows, or if any column has fewer than two unique
        non-null levels.
    """
    # Empty selections never reach here: the recipe detects a zero-column
    # selection and treats the step as the identity transform, so ``fit`` is
    # not called. See ``Recipe.fit`` / ``Step._empty_selection``.
    if selected.shape[0] == 0:
        raise ValueError(
            "cannot fit one-hot encoding on a zero-row frame; no levels can be learned"
        )
    result: dict[str, list[Any]] = {}
    for col in selected.columns:
        levels: list[Any] = selected[col].drop_nulls().unique().sort().to_list()
        if len(levels) < 2:
            raise ValueError(
                f"column {col!r} has fewer than 2 unique non-null levels "
                f"({len(levels)} found); one-hot encoding requires at least 2 levels"
            )
        result[col] = levels
    return result


class StepOneHot(Step):
    """Expand discrete columns into *k* binary indicator columns.

    Each selected column is replaced by *k* indicator columns where *k* is the
    number of unique non-null levels seen during ``fit()``. Indicators are
    ``nw.Int8`` (0/1); a column that contains null-source or unseen-level rows
    is ``nw.Float64`` instead (0.0/1.0/NaN), since pandas' ``Int8`` cannot hold
    the missing values. Unlike `StepDummy`, no reference level is dropped.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to expand. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.nominal_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.string()``, ``polars.selectors.string()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only discrete
        columns are meaningful here.
    name_fn
        Callable mapping ``(source_column, level_as_str)`` to the indicator
        column's name. Default ``None``, which names them ``"{column}_{level}"``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the indicators
        instead of being dropped. Default ``False``.
    id
        Stable step identifier. Defaults to ``"one_hot"``; a second step of
        this type in the same recipe becomes ``"one_hot_2"``, and passing an
        ``id`` already used in the recipe is an error.

    Examples
    --------
    Each categorical column is replaced by one indicator per level, with no
    reference level dropped (contrast `StepDummy`):

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepOneHot
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepOneHot(s.nominal_features()))
    recipe
    ```

    `fit()` records the levels of each selected column:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the column with one indicator per level:

    ```{python}
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside them:

    ```{python}
    recipe = Recipe().add(StepOneHot(s.nominal_features(), keep_original_cols=True))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: Selector,
        *,
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.name_fn = name_fn
        super().__init__(
            id, "one_hot", keep_original_cols=keep_original_cols, output_role="feature"
        )
        self._estimates: dict[str, list[Any]] | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the full sorted level list for each selected column."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_one_hot(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        produced, self._output_parents = dummy_output_parents(
            self._estimates, self._col_name
        )
        self._schema_out = ColSchema(columns=produced)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply stored level lists to produce indicator columns."""
        if self._estimates is None:
            raise RuntimeError("StepOneHot.transform() called before fit()")
        return apply_dummy(
            new_data, self._estimates, self._col_name, self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepOneHot.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the indicator columns this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepOneHot.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each indicator column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepOneHot.output_parents() called before fit()")
        return self._output_parents

    def produced_columns(self) -> list[str]:
        """Indicator names in order, preserving duplicates (see ``Step``)."""
        return list(self.schema_out().columns)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Accepts discrete columns only (see ``DISCRETE_DTYPES``)."""
        return set(DISCRETE_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._estimates is None:
            return
        dummy_update_roles(
            roles,
            self._estimates,
            self._col_name,
            keep_original_cols=self.keep_original_cols,
        )

    @property
    def produces_sparse(self) -> bool:
        """``StepOneHot`` never produces sparse output."""
        return False

    def _col_name(self, col: str, level: str) -> str:
        return resolve_col_name(self.name_fn, col, level)
