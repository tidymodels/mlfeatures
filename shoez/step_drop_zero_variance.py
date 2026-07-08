"""Zero-variance drop: find columns with a single unique value.

The *fit* half of the zero-variance rule lives in `fit_drop_zero_variance()`
as a plain function so the logic can be exercised on its own;
`StepDropZeroVariance` wraps it in the `Step`
lifecycle so a `Recipe` can fit and apply it.

The *apply* half (dropping the named columns) is `shoez._frame.apply_drop()`,
shared with every REMOVING-mode step (``step_drop_zv``, ``step_drop_corr``, ...).
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


def fit_drop_zero_variance(features: Frame) -> list[str]:
    """Find columns with a single unique non-null value.

    A column is "zero-variance" if every observed (non-null) value is the
    same. Missing values are ignored, so ``[5, 5, NaN]`` counts as
    zero-variance (one observed value). A column whose values are *all*
    missing has zero observed values and is also treated as zero-variance.

    Parameters
    ----------
    features
        Internal `Frame` (narwhals) holding the candidate
        predictor columns. Native frames are wrapped via
        `shoez._frame.from_native()` at the recipe boundary.

    Returns
    -------
    list[str]
        Column names to drop, in their input order. Pair with
        `shoez._frame.apply_drop()` to remove the offending columns
        from any frame with the same schema.
    """
    return [c for c in features.columns if features[c].drop_nulls().n_unique() <= 1]


class StepDropZeroVariance(Step):
    """Remove columns that have a single unique value (zero variance).

    A ``REMOVING`` step: the selected columns whose observed values are all the
    same — or all missing — are dropped, and no new columns are produced. The
    decision is made by `fit_drop_zero_variance()`; this class wraps it in
    the `Step` lifecycle so a `Recipe`
    can fit and apply it.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.

    Examples
    --------
    ``const`` has a single unique value, so it is dropped; ``x`` is kept:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepDropZeroVariance
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "const": [5.0, 5.0, 5.0]})
    Recipe().add(StepDropZeroVariance(["x", "const"])).fit(df).transform(df)
    ```
    """

    mode = Mode.REMOVING

    def __init__(
        self,
        columns: Selector,
        id: str | None = None,
    ):
        self.columns = columns
        self.id = id if id is not None else "zv"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``.
        self._estimates: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate which selected columns to drop and store the decision."""
        features = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_drop_zero_variance(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
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
