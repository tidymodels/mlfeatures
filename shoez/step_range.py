"""Min-max scaling: fit per-column bounds, then rescale to a fixed range.

The *fit* half lives in `fit_range()` and the *apply* half in `apply_range()`,
both plain functions so the data-ops core can be exercised on its own.
`StepRange` wraps them in the `Step` lifecycle so a `Recipe` can fit and apply
it.

They are the pure ``prep`` / ``bake`` counterparts of R ``recipes``'
``step_range``: the bounds are estimated once at fit time and reused unchanged
at transform time as ``(x - lo) / (hi - lo) * (max - min) + min``.
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

__all__ = ["fit_range", "apply_range", "StepRange"]


def fit_range(
    selected: Frame,
) -> dict[str, tuple[float, float]]:
    """Estimate per-column lower and upper bounds for min-max scaling.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.

    Callers are expected to pass an all-numeric frame; dtype screening is the
    selector / ``accepts_dtypes`` layer's job, so this function does not
    re-validate it.

    Returns
    -------
    ``dict`` mapping each rescalable column name to its ``(lo, hi)`` pair.
    Empty when ``selected`` has no columns.

    Columns whose range ``hi - lo`` is ``0`` (constant columns — every value the
    same) cannot be spread across the target range, so they are **omitted** from
    the result and a `UserWarning` is emitted naming them.

    Notes
    -----
    Based on the ``prep`` step of ``step_range`` in the R ``recipes`` package.
    A ``NaN`` bound (e.g. from a zero-row fit) is not treated as constant — it
    is kept and propagates as ``NaN`` at transform time, signalling a degenerate
    fit.
    """
    bounds: dict[str, tuple[float, float]] = {}
    constant: list[str] = []

    for col in selected.columns:
        lo = as_float(selected[col].min())
        hi = as_float(selected[col].max())
        if hi - lo == 0.0:
            constant.append(col)
            continue
        bounds[col] = (lo, hi)

    if constant:
        warnings.warn(
            f"{len(constant)} column(s) had a zero range (constant) "
            f"and were not rescaled: {constant!r}",
            stacklevel=2,
        )

    return bounds


def apply_range(
    selected: Frame,
    bounds: dict[str, tuple[float, float]],
    *,
    min: float = 0.0,
    max: float = 1.0,
    clipping: bool = True,
) -> Frame:
    """Rescale ``selected`` into ``[min, max]`` using stored bounds.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to rescale.
    bounds:
        The ``{column: (lo, hi)}`` mapping returned by `fit_range()`. Any column
        of ``selected`` not present as a key (e.g. a constant column dropped at
        fit time) is passed through untouched.
    min:
        Lower end of the target range.
    max:
        Upper end of the target range.
    clipping:
        When true, values outside the fitted ``[lo, hi]`` are clamped into
        ``[min, max]`` rather than allowed to land outside it.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``bounds`` replaced
    in place by ``(x - lo) / (hi - lo) * (max - min) + min``. Column names,
    order, and structure are unchanged.
    """
    exprs: list[nw.Expr] = []
    for col, (lo, hi) in bounds.items():
        scaled = (nw.col(col) - lo) / (hi - lo) * (max - min) + min
        if clipping:
            scaled = scaled.clip(min, max)
        exprs.append(scaled.alias(col))
    return selected.with_columns(*exprs)


class StepRange(Step):
    """Scale numeric columns to a fixed range using training-set bounds.

    Each selected numeric column ``x`` is replaced in place by
    ``(x - lo) / (hi - lo) * (max - min) + min``, where ``lo`` and ``hi`` are
    the column's smallest and largest values in the training data. On the
    training data itself the result therefore spans exactly ``min`` to ``max``.

    New values falling outside the training bounds would land outside the target
    range; by default they are clamped back into it, which ``clipping=False``
    turns off.

    Constant columns among the selection have a zero range and cannot be spread
    across the target range; they are dropped from the rescaling (with a
    warning) and pass through ``transform`` untouched.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to rescale. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    min
        Lower end of the target range. Defaults to ``0.0``.
    max
        Upper end of the target range. Defaults to ``1.0``, and must be greater
        than ``min``.
    clipping
        Whether values outside the training bounds are clamped into
        ``[min, max]`` at transform time. Defaults to ``True``.
    id
        Stable step identifier. Defaults to ``"range"``; a second step of this
        type in the same recipe becomes ``"range_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its rescaled values:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepRange
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 5.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepRange(s.numeric_features()))
    recipe
    ```

    `fit()` learns each column's smallest and largest value:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` applies them in place:

    ```{python}
    fitted.transform(df)
    ```

    ``min`` and ``max`` set the target range:

    ```{python}
    Recipe().add(StepRange(s.numeric_features(), min=-1.0, max=1.0)).fit(
        df
    ).transform(df)
    ```

    Values beyond the training bounds are clamped, unless ``clipping=False``
    lets them extrapolate:

    ```{python}
    new_df = pd.DataFrame({"x": [0.0, 9.0]})
    Recipe().add(StepRange(s.numeric_features(), clipping=False)).fit(
        df
    ).transform(new_df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        *,
        min: float = 0.0,
        max: float = 1.0,
        clipping: bool = True,
        id: str | None = None,
    ):
        if min >= max:
            raise ValueError(
                f"'min' must be less than 'max'; got min={min!r} and max={max!r}"
            )
        self.columns = columns
        self.min = min
        self.max = max
        self.clipping = clipping
        super().__init__(id, "range")
        # Populated by ``fit()``: {column: (lo, hi)} over rescalable columns.
        self._estimates: dict[str, tuple[float, float]] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate per-column lower and upper bounds over the selected columns."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_range(selected)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply the stored bounds to rescale into the target range."""
        if self._estimates is None:
            raise RuntimeError("StepRange.transform() called before fit()")
        return apply_range(
            new_data,
            self._estimates,
            min=self.min,
            max=self.max,
            clipping=self.clipping,
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepRange.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, rescaled ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepRange.schema_out() called before fit()")
        rescaled = set(self._estimates)
        dtypes = {
            col: ("Float64" if col in rescaled else self._schema_in.dtypes[col])
            for col in self._schema_in.columns
        }
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Min-max scaling is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """Shifting the low bound onto ``min`` moves zeros; output stays dense."""
        return False
