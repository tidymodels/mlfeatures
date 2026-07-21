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


def apply_filter(features: Frame, predicate: Predicate) -> Frame:
    """Keep only the rows for which ``predicate`` evaluates to ``True``.

    Parameters
    ----------
    features
        Internal `Frame` (narwhals). Native frames are wrapped via
        `shoez._frame.from_native()` at the recipe boundary.
    predicate
        Callable taking ``features`` and returning a boolean narwhals
        expression or series, ``True`` for the rows to retain (``dplyr::filter``
        / ``pl.DataFrame.filter`` semantics). The mask is built from the frame
        it is applied to, so it works unchanged across backends and on 0-row and
        1-row frames.

    Returns
    -------
    Frame
        ``features`` with the non-matching rows removed. The column schema is
        never touched — this is a pure row operation.
    """
    return features.filter(predicate(features))


class StepFilter(RowOp):
    """Remove rows that do not match a predicate.

    A ``ROW_OP`` step: it changes the number of rows but never the column
    schema. Because it is a `RowOp`, it only takes effect on the training frame
    during ``fit()``; the framework skips it at ``transform()`` time so rows are
    never dropped from data being scored. The row selection is performed by
    `apply_filter()`; this class wraps it in the `RowOp` lifecycle so a `Recipe`
    can fit and apply it.

    Being a row operation, it consumes no *named* columns and produces none, so
    there is nothing to estimate: ``fit()`` only records the (unchanged) schema.

    Parameters
    ----------
    predicate
        Callable taking the frame and returning a boolean narwhals expression or
        series, ``True`` for the rows to keep. Must be callable.
    id
        Stable step identifier. Default ``"filter"``.

    Examples
    --------
    Rows where ``age`` is 30 or below are dropped at fit time; ``transform()``
    on new data leaves every row in place (a `RowOp` never filters what it is
    scoring):

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepFilter
    ```

    ```{python}
    df = pd.DataFrame({"age": [25, 41, 33, 52]})
    Recipe().add(StepFilter(lambda df: df["age"] > 30)).fit(df).transform(df)
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

        ``StepFilter`` is a `RowOp`: its ``transform`` is a no-op on new data,
        so the predicate is never exercised once fitted. It is a user-supplied
        callable (not plain data), so it is dropped from the JSON payload rather
        than trying to serialize a closure. A recipe reloaded from JSON filters
        no rows at ``transform`` -- which is already the `RowOp` contract.
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

        The framework never calls this on new data (``StepFilter`` is a
        `RowOp`); it runs only on the training frame at fit time.
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
