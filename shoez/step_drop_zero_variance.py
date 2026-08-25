"""Zero-variance drop: find columns with a single unique value.

The *fit* half of the zero-variance rule lives in `fit_drop_zero_variance()`
as a plain function so the logic can be exercised on its own;
`StepDropZeroVariance` wraps it in the `Step`
lifecycle so a `Recipe` can fit and apply it.

The *apply* half (dropping the named columns) is `shoez._frame.apply_drop()`,
which skips names that are absent from the frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from shoez import _frame
from shoez.step import Step
from shoez.types import (
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

__all__ = ["fit_drop_zero_variance", "StepDropZeroVariance"]


def fit_drop_zero_variance(selected: Frame) -> list[str]:
    """Find columns with a single unique non-null value.

    A column is "zero-variance" if every observed (non-null) value is the
    same. Missing values are ignored, so ``[5, 5, NaN]`` counts as
    zero-variance (one observed value). A column whose values are *all*
    missing has zero observed values and is also treated as zero-variance.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the candidate
        feature columns. Native frames are wrapped via
        `shoez._frame.from_native()` at the recipe boundary.

    Returns
    -------
    list[str]
        Column names to drop, in their input order. Pair with
        `shoez._frame.apply_drop()` to remove the offending columns
        from any frame with the same schema.
    """
    return [c for c in selected.columns if selected[c].drop_nulls().n_unique() <= 1]


class StepDropZeroVariance(Step):
    """Remove columns that have a single unique value (zero variance).

    The selected columns whose observed values are all the same — or all
    missing — are dropped, and no new columns are produced.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to consider for removal. Accepts a single column name, a list
        of names, a shoez selector such as ``Selectors.features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Any dtype is
        accepted.
    id
        Stable step identifier. Defaults to ``"zv"``; a second step of this
        type in the same recipe becomes ``"zv_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    ``const`` has a single unique value, so it is dropped; ``x`` is kept:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepDropZeroVariance
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "const": [5.0, 5.0, 5.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDropZeroVariance(s.numeric_features()))
    recipe
    ```

    `fit()` counts the unique values in each column:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` drops the constant one:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.REMOVING

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        self.columns = columns
        self._init_id(id, "zv")
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``.
        self._estimates: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate which selected columns to drop and store the decision."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_drop_zero_variance(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Drop the columns chosen at fit time. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepDropZeroVariance.transform() called before fit()")
        return _frame.apply_drop(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDropZeroVariance.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the columns this step produced — none; it only removes."""
        if self._estimates is None:
            raise RuntimeError("StepDropZeroVariance.schema_out() called before fit()")
        return ColSchema(columns=[])

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Zero variance is meaningful for any dtype, not just numerics."""
        return ANY_DTYPE

    def update_roles(self, roles: Roles) -> None:
        if self._estimates:
            for col in self._estimates:
                for role in list(roles.of(col)):
                    roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """``step_drop_zero_variance`` never produces sparse output."""
        return False
