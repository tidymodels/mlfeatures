"""Zero-variance drop: find columns with a single unique value.

The *fit* half of the zero-variance rule lives in `fit_drop_zero_variance()`
as a plain function so the logic can be exercised on its own (and reused by
`shoez.step_drop_near_zero_variance`, whose rule is a superset of this one);
`StepDropZeroVariance` supplies it as the `_drop_candidates()` rule for the
fit/transform/schema lifecycle shared with the other ``step_drop_*`` steps in
`shoez._drop.ColumnDropStep`.
"""

from __future__ import annotations

from shoez._drop import ColumnDropStep
from shoez.types import ANY_DTYPE, AcceptedDtypes, Frame, Selector

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


class StepDropZeroVariance(ColumnDropStep):
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

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        super().__init__(columns, id=id, default_id="zv")

    def _drop_candidates(self, selected: Frame) -> list[str]:
        return fit_drop_zero_variance(selected)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Zero variance is meaningful for any dtype, not just numerics."""
        return ANY_DTYPE
