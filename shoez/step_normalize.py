"""Center-and-scale normalization: fit per-column mean/std, then standardize.

The *fit* half lives in `fit_normalize()` and the *apply* half in
`apply_normalize()`, both plain functions so the data-ops core can be
exercised on its own. `StepNormalize` wraps them in the
`Step` lifecycle so a `Recipe` can fit
and apply it.

They are the pure ``prep`` / ``bake`` counterparts of R ``recipes``'
``step_normalize``: the statistics are estimated once at fit time and reused
unchanged at transform time as ``(x - mean) / std``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez._frame import as_float
from shoez.step import Step
from shoez.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_normalize", "apply_normalize", "StepNormalize"]


def fit_normalize(
    features: Frame,
) -> dict[str, tuple[float, float]]:
    """Estimate per-column mean and standard deviation for normalization.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Callers are expected to pass an all-numeric frame; dtype screening is the
    selector / ``accepts_dtypes`` layer's job, so this function does not
    re-validate it.

    Returns
    -------
    ``dict`` mapping each normalizable column name to its ``(mean, std)`` pair.
    Empty when ``features`` has no columns.

    Columns with a standard deviation of ``0`` (constant columns — every value
    the same) cannot be scaled to unit variance, so they are **omitted** from
    the result and a `UserWarning` is emitted naming them.

    Notes
    -----
    Based on the ``prep`` step of ``step_normalize`` in the R ``recipes``
    package.  Uses the sample standard deviation (``ddof=1``, pinned explicitly),
    matching R's ``sd()``; scaling by it makes the normalized column's sample
    standard deviation exactly ``1``.  A ``NaN`` standard deviation (e.g. from a
    single-row fit) is not treated as constant — it is kept and propagates as
    ``NaN`` at transform time, signalling a degenerate fit.
    """
    stats: dict[str, tuple[float, float]] = {}
    constant: list[str] = []

    for col in features.columns:
        # ddof=1 (sample std) explicitly: scaling by it makes the sample std of
        # the normalized column exactly 1, matching R's sd().
        std = as_float(features[col].std(ddof=1))
        if std == 0.0:
            constant.append(col)
            continue
        stats[col] = (as_float(features[col].mean()), std)

    if constant:
        warnings.warn(
            f"{len(constant)} column(s) had zero standard deviation (constant) "
            f"and were not normalized: {constant!r}",
            stacklevel=2,
        )

    return stats


def apply_normalize(
    features: Frame,
    stats: dict[str, tuple[float, float]],
) -> Frame:
    """Standardize ``features`` as ``(x - mean) / std`` using stored statistics.

    Parameters
    ----------
    features:
        A narwhals DataFrame to normalize.
    stats:
        The ``{column: (mean, std)}`` mapping returned by `fit_normalize()`. Any
        column of ``features`` not present as a key (e.g. a constant column
        dropped at fit time) is passed through untouched.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``stats`` replaced
    in place by its standardized value. Column names, order, and structure are
    unchanged.
    """
    exprs = [
        ((nw.col(col) - mean) / std).alias(col) for col, (mean, std) in stats.items()
    ]
    return features.with_columns(*exprs)


class StepNormalize(Step):
    """Center and scale numeric columns to zero mean and unit variance.

    Each selected numeric column ``x`` is replaced in place by
    ``(x - mean) / std``, with the mean and standard deviation estimated at fit
    time and reused unchanged at transform time.

    Constant (zero-variance) columns among the selection cannot be scaled to
    unit variance; they are dropped from the normalization (with a warning) and
    pass through ``transform`` untouched.

    Examples
    --------
    Each selected column is replaced in place by its centered, unit-variance
    values:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepNormalize
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepNormalize(s.numeric_predictors()))
    recipe
    ```

    `fit()` learns each column's mean and standard deviation:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` applies them in place:

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
        self.id = id if id is not None else "normalize"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``: {column: (mean, std)} over normalizable columns.
        self._estimates: dict[str, tuple[float, float]] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate per-column mean and std over the selected columns."""
        features = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_normalize(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply the stored mean and std: ``(x - mean) / std``. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepNormalize.transform() called before fit()")
        return apply_normalize(new_data, self._estimates)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepNormalize.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, normalized ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepNormalize.schema_out() called before fit()")
        normalized = set(self._estimates)
        dtypes = {
            col: ("Float64" if col in normalized else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Normalization is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Centering shifts zeros off zero, but output stays dense."""
        return False
