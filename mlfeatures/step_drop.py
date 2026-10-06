"""Unconditional column removal: drop whatever the selector resolved to.

Unlike the other ``step_drop_*`` modules, there is nothing to estimate here, so
this one has no ``fit_*`` half: the resolved selection *is* the state.
``StepDrop``'s only job is `_drop_candidates()`, returning every resolved
column; the fit/transform/schema lifecycle shared with the other
``step_drop_*`` steps lives in `mlfeatures._drop.ColumnDropStep`.
"""

from __future__ import annotations

from mlfeatures._drop import ColumnDropStep
from mlfeatures.types import ANY_DTYPE, AcceptedDtypes, Frame, Selector

__all__ = ["StepDrop"]


class StepDrop(ColumnDropStep):
    """Remove the selected columns.

    Every column the selector matches is dropped, and no new columns are
    produced. Nothing is estimated: which columns to drop is decided by the
    selector alone, fixed against the training frame's schema.

    The selection is taken at face value, so a selector that reaches a column
    with a role other than ``"feature"`` drops that column too.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to remove. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.features()`` (see `Selectors` for the
        full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Any dtype is
        accepted.
    id
        Stable step identifier. Defaults to ``"drop"``; a second step of this
        type in the same recipe becomes ``"drop_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    It is often easier to say what to remove than to enumerate what to keep:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepDrop
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "note": ["a", "b", "c"]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDrop(s.string_features()))
    recipe
    ```

    `fit()` resolves the selector against the training schema:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` drops the columns it matched:

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
        super().__init__(columns, id=id, default_id="drop")

    def _drop_candidates(self, selected: Frame) -> list[str]:
        """Every resolved column is dropped; the selector alone decides."""
        return list(selected.columns)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Dropping a column is meaningful whatever its dtype."""
        return ANY_DTYPE
