"""Row filter: keep only the rows matching a predicate.

`step_filter` is the ``ROW_OP`` prototype from ``design/steps-catalog.md``,
modelled on ``step_filter`` from the R ``recipes`` package. It is a `RowOp`, so
it only ever takes effect on the **training** frame during ``fit()`` — the
framework skips it structurally at ``transform()`` time on new data (see
``design/step-contract.md``). This is exactly what "exercises skip-at-transform"
means: a filter must not drop rows from the data being scored.

The *apply* half lives in `apply_filter()` as a plain function so the row
selection can be exercised on its own; `StepFilter` wraps it in the `RowOp`
lifecycle so a `Recipe` can fit and apply it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from shoez import _frame
from shoez.step import RowOp
from shoez.types import (
    ANY_DTYPE,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["apply_filter", "StepFilter"]

#: A predicate maps the internal `Frame` to a boolean mask (a narwhals
#: expression or series) that is ``True`` for the rows to **keep**.
Predicate = Callable[[Frame], Any]


def _keep_all_rows(frame: Frame) -> bool:
    """A predicate that keeps every row; the inert default a deserialized filter carries."""
    return True


def apply_filter(selected: Frame, predicate: Predicate) -> Frame:
    """Keep only the rows for which ``predicate`` evaluates to ``True``.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals). Native frames are wrapped via
        `shoez._frame.from_native()` at the recipe boundary.
    predicate
        Callable taking ``selected`` and returning a boolean narwhals
        expression or series, ``True`` for the rows to retain (``dplyr::filter``
        / ``pl.DataFrame.filter`` semantics). The mask is built from the frame
        it is applied to, so it works unchanged across backends and on 0-row and
        1-row frames.

    Returns
    -------
    Frame
        ``selected`` with the non-matching rows removed. The column schema is
        never touched — this is a pure row operation.
    """
    return selected.filter(predicate(selected))


class StepFilter(RowOp):
    """Remove rows that do not match a predicate.

    Rows for which ``predicate`` is not ``True`` are dropped.

    Filtering applies to the training data only. Fitting a recipe drops the
    non-matching training rows, but transforming new data returns every row
    given to it, because dropping rows from data being scored would silently
    lose predictions. So a recipe that filters is not a way to filter new data.

    Parameters
    ----------
    predicate
        Callable taking the frame and returning a boolean narwhals expression or
        series, ``True`` for the rows to keep. Must be callable.
    id
        Stable step identifier. Defaults to ``"filter"`` when not given; the
        default is per step type, so two steps of this type in one recipe share
        it unless you set explicit ids.

    Examples
    --------
    Rows where ``age`` is 30 or below are dropped when the recipe is fitted,
    while ``transform()`` leaves every row in place:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepFilter
    ```

    ```{python}
    df = pd.DataFrame({"age": [25, 41, 33, 52]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepFilter(lambda df: df["age"] > 30))
    recipe
    ```

    `fit()` applies the predicate to the training data:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` leaves new data untouched, so all four rows come back:

    ```{python}
    fitted.transform(df)
    ```
    """

    mode = Mode.ROW_OP

    def __init__(
        self,
        predicate: Predicate,
        id: str | None = None,
    ) -> None:
        if not callable(predicate):
            raise TypeError(
                f"'predicate' must be callable; got {type(predicate).__name__!r}"
            )
        self.predicate = predicate
        self.id = id if id is not None else "filter"
        self.trained = False
        # A row operation touches no columns, so keeping originals is the only
        # sensible setting; there is nothing to drop.
        self.keep_original_cols = True
        self.output_role = None
        # Populated by ``fit()``. A filter changes no columns, so ``schema_in``
        # and ``schema_out`` are the same frame schema.
        self._schema: ColSchema | None = None

    def to_state(self) -> dict[str, object]:
        """Serialize, omitting the row predicate.

        Because filtering never applies to new data, the predicate is never
        exercised once the recipe is fitted. It is a user-supplied callable
        rather than plain data, so it is dropped from the JSON payload instead
        of trying to serialize a closure. A recipe reloaded from JSON therefore
        behaves exactly as one that was never serialized: it filters no rows at
        ``transform`` time.
        """
        from shoez.serialization import _encode  # pyright: ignore[reportPrivateUsage]

        return {k: _encode(v) for k, v in self.__dict__.items() if k != "predicate"}

    @classmethod
    def from_state(cls, state: dict[str, object]) -> "Self":
        """Reconstruct a transform-only filter; the dropped predicate is inert."""
        step = super().from_state(state)
        if not hasattr(step, "predicate"):
            # A RowOp never runs transform() on new data, so this no-op predicate
            # is never invoked; it exists only so attribute access stays defined.
            step.predicate = _keep_all_rows
        return step

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record the (unchanged) column schema. Estimates nothing."""
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(training_data).items()}
        self._schema = ColSchema(columns=_frame.columns(training_data), dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Keep only the rows matching the predicate.

        The framework calls this only on the training frame while fitting;
        transforming new data never filters rows.
        """
        if self._schema is None:
            raise RuntimeError("StepFilter.transform() called before fit()")
        return apply_filter(new_data, self.predicate)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed — the frame passes through."""
        if self._schema is None:
            raise RuntimeError("StepFilter.schema_in() called before fit()")
        return self._schema

    def schema_out(self) -> ColSchema:
        """Schema this step produced — identical to the input; only rows change."""
        if self._schema is None:
            raise RuntimeError("StepFilter.schema_out() called before fit()")
        return self._schema

    def accepts_dtypes(self) -> AcceptedDtypes:
        """A row filter is meaningful regardless of column dtype."""
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        """``step_filter`` never produces sparse output."""
        return False
