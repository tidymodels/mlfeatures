"""Unconditional column removal: drop whatever the selector resolved to.

Unlike the other ``step_drop_*`` modules, there is nothing to estimate here, so
this one has no ``fit_*`` half: the resolved selection *is* the state. ``fit()``
records which columns the selector matched on the training frame, and the
*apply* half is `shoez._frame.apply_drop()`, which skips names that are absent
from the frame.
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

__all__ = ["StepDrop"]


class StepDrop(Step):
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
        shoez selector such as ``Selectors.features()`` (see `Selectors` for the
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
    from shoez import Recipe, StepDrop
    from shoez import Selectors as s
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

    mode = Mode.REMOVING

    def __init__(
        self,
        columns: Selector,
        *,
        id: str | None = None,
    ):
        self.columns = columns
        super().__init__(id, "drop")
        # Populated by ``fit()``: the resolved names, which are what gets dropped.
        self._estimates: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the selector and store the columns it matched."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = list(selected.columns)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Drop the columns chosen at fit time. Never re-resolves the selector."""
        if self._estimates is None:
            raise RuntimeError("StepDrop.transform() called before fit()")
        return _frame.apply_drop(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDrop.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the columns this step produced — none; it only removes."""
        if self._estimates is None:
            raise RuntimeError("StepDrop.schema_out() called before fit()")
        return ColSchema(columns=[])

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Dropping a column is meaningful whatever its dtype."""
        return ANY_DTYPE

    def update_roles(self, roles: Roles) -> None:
        if self._estimates:
            for col in self._estimates:
                for role in list(roles.of(col)):
                    roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """``step_drop`` never produces sparse output."""
        return False
