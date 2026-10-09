"""Logarithmic transform: log the selected numeric columns elementwise.

The *fit* half lives in `fit_log()` and the *apply* half in `apply_log()`, both
plain functions so the data-ops core can be exercised on its own. `StepLog`
wraps them in the `Step` lifecycle so a `Recipe` can fit and apply it.

They are the pure ``prep`` / ``bake`` counterparts of R ``recipes``'
``step_log``: nothing statistical is estimated, so the fit half only resolves
which columns to touch and screens them for values outside the logarithm's
domain, while the apply half computes ``log(x + offset) / log(base)``.

``signed`` selects the other formula recipes offers, ``sign(x) * log(|x|)``,
with ``|x| < 1`` flattened to ``0`` (`_signed_log_expr`). The flattening is what
makes it continuous through the origin: without it, ``|x|`` just below ``1``
would log to a large negative number on both sides of zero. It is R
``recipes``' rule too, and it means the whole real line is in the domain, so the
warning below never fires under ``signed``.

Unlike the other numeric steps, the domain screen runs at *both* fit and
transform time (`_warn_non_positive`): a column can be strictly positive in the
training data and still contain a zero or a negative in new data, and that is
exactly the case worth warning about. It costs one ``min()`` per column per
call.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures._frame import as_float
from mlfeatures._numeric_one_to_one import schema_out_for_touched
from mlfeatures.step import Step
from mlfeatures.types import (
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

__all__ = ["fit_log", "apply_log", "StepLog"]


def _warn_non_positive(
    selected: Frame, columns: list[str], *, offset: float, stacklevel: int = 3
) -> None:
    """Warn about columns holding values outside the logarithm's domain.

    A value ``v`` is shifted to ``v + offset`` before the logarithm, so the
    domain boundary is ``-offset``: ``v == -offset`` yields ``-inf`` and
    ``v < -offset`` yields ``NaN``. Both are legal outputs (R ``recipes`` lets
    them through too), so this only warns; nothing is dropped or clamped.

    A column whose minimum is missing (all-null, or zero rows) has a ``NaN``
    minimum, which compares false and therefore does not warn.
    """
    offending = [col for col in columns if as_float(selected[col].min()) <= -offset]
    if offending:
        warnings.warn(
            f"{len(offending)} column(s) had value(s) at or below the negated "
            f"'offset'; log will produce -inf or NaN: {offending!r}",
            stacklevel=stacklevel,
        )


def _signed_log_expr(col: str, *, base: float) -> nw.Expr:
    """Build ``sign(x) * log(|x|, base)``, with ``|x| < 1`` flattened to ``0``.

    narwhals has no ``Expr.sign()``, so the sign is spelled out as a ``when``.
    The ``|x| < 1`` branch also swallows ``x == 0``, which would otherwise log
    to ``-inf``.
    """
    x = nw.col(col)
    sign = nw.when(x < 0).then(nw.lit(-1.0)).otherwise(nw.lit(1.0))
    return (
        nw.when(x.abs() < 1)
        .then(nw.lit(0.0))
        .otherwise(sign * x.abs().log(base=base))
        .alias(col)
    )


def fit_log(selected: Frame, *, offset: float = 0.0, signed: bool = False) -> list[str]:
    """Resolve which columns the log transform will touch.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all numeric. Passed in by the
        step's ``fit()`` method after selector resolution.
    offset:
        The shift applied before the logarithm, used here only to decide which
        columns fall outside its domain.
    signed:
        When true, the signed formula is used instead, which is defined for
        every real value; the domain screen is then skipped.

    Callers are expected to pass an all-numeric frame; dtype screening is the
    selector / ``accepts_dtypes`` layer's job, so this function does not
    re-validate it.

    Returns
    -------
    ``list`` of the column names to transform, in frame order. Empty when
    ``selected`` has no columns.

    Nothing is estimated from the data: the logarithm is elementwise, so the
    resolved column list *is* the whole fitted state. Columns holding values at
    or below ``-offset`` are kept (they log to ``-inf`` or ``NaN``, which is
    informative output rather than an error) but a `UserWarning` names them.
    """
    columns = list(selected.columns)
    if not signed:
        _warn_non_positive(selected, columns, offset=offset)
    return columns


def apply_log(
    selected: Frame,
    columns: list[str],
    *,
    base: float = math.e,
    offset: float = 0.0,
    signed: bool = False,
) -> Frame:
    """Replace each column in ``columns`` by ``log(x + offset) / log(base)``.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to transform.
    columns:
        The column names returned by `fit_log()`. Any other column of
        ``selected`` is passed through untouched.
    base:
        Base of the logarithm.
    offset:
        Added to every value before the logarithm. Ignored under ``signed``,
        whose formula takes no offset.
    signed:
        When true, compute ``sign(x) * log(|x|, base)`` with ``|x| < 1``
        flattened to ``0`` instead (see `_signed_log_expr`).

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``columns``
    replaced in place by its logarithm. Column names, order, and structure are
    unchanged.

    Warns about values outside the logarithm's domain the same way `fit_log()`
    does, because new data can leave it even when the training data did not. The
    numpy ``RuntimeWarning``s pandas raises for those values ("divide by zero
    encountered in log") are suppressed so both backends behave alike and that
    single `UserWarning` is the only signal. Under ``signed`` nothing is out of
    domain, so no warning is possible; the suppression still applies, because
    pandas evaluates the ``|x| < 1`` branch's ``log(0)`` before discarding it.
    """
    exprs: list[nw.Expr]
    if signed:
        exprs = [_signed_log_expr(col, base=base) for col in columns]
    else:
        _warn_non_positive(selected, columns, offset=offset)
        exprs = [(nw.col(col) + offset).log(base=base).alias(col) for col in columns]
    with warnings.catch_warnings():
        # polars computes this silently; pandas routes through numpy, which
        # warns on log(0) and log(negative). The UserWarning above already
        # reported them, and it was emitted before this block.
        warnings.simplefilter("ignore", RuntimeWarning)
        return selected.with_columns(*exprs)


class StepLog(Step):
    """Take the logarithm of numeric columns.

    Each selected numeric column ``x`` is replaced in place by
    ``log(x + offset) / log(base)``. Nothing is estimated from the training
    data: the transform is elementwise, so it is identical on training and new
    data.

    The logarithm is undefined at and below ``-offset``: a value equal to
    ``-offset`` becomes ``-inf`` and anything smaller becomes ``NaN``. Those
    values pass through rather than raising, and a warning names the columns
    they appear in. Because new data can hold such values even when the
    training data did not, the check runs at fit *and* transform time. Setting
    ``offset`` to a positive number is the usual way to bring zeros into the
    domain.

    ``signed`` is the other way to handle a column that spans zero. It computes
    ``sign(x) * log(|x|)`` instead, flattening ``|x| < 1`` to ``0`` so the
    result is continuous through the origin. Every real value is then in the
    domain and no warning is possible, at the cost of a transform that is flat
    across the interval from ``-1`` to ``1``. ``offset`` takes no part in that
    formula, so passing both is an error rather than a silently ignored
    argument.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to log-transform. Accepts a single column name, a list of names,
        a mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    base
        Base of the logarithm. Defaults to ``math.e`` (the natural logarithm),
        and must be positive and not ``1``.
    offset
        Added to every value before the logarithm, which shifts what counts as
        out of domain. Defaults to ``0.0``. Cannot be combined with ``signed``.
    signed
        Whether to compute ``sign(x) * log(|x|)``, with ``|x| < 1`` flattened to
        ``0``, instead of the plain logarithm. Defaults to ``False``.
    id
        Stable step identifier. Defaults to ``"log"``; a second step of this
        type in the same recipe becomes ``"log_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    Each selected column is replaced in place by its logarithm:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepLog
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"x": [1.0, 10.0, 100.0, 1000.0]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepLog(s.numeric_features()))
    recipe
    ```

    There is nothing to estimate, so `fit()` only resolves the selection:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` takes the natural logarithm:

    ```{python}
    fitted.transform(df)
    ```

    ``base`` changes the logarithm's base, here to the base-10 logarithm:

    ```{python}
    Recipe().add(StepLog(s.numeric_features(), base=10)).fit(df).transform(df)
    ```

    Zeros are outside the logarithm's domain:

    ```{python}
    counts = pd.DataFrame({"x": [0.0, 1.0, 3.0, 7.0]})
    counts
    ```

    ``offset`` shifts every value up first, which brings them back in:

    ```{python}
    Recipe().add(StepLog(s.numeric_features(), offset=1.0)).fit(
        counts
    ).transform(counts)
    ```

    A column spanning zero has negatives too, which no offset would fix without
    distorting the large values:

    ```{python}
    spread = pd.DataFrame({"x": [-100.0, -1.0, 0.0, 1.0, 100.0]})
    spread
    ```

    ``signed`` logs each side on its own scale and flattens the middle:

    ```{python}
    Recipe().add(StepLog(s.numeric_features(), base=10, signed=True)).fit(
        spread
    ).transform(spread)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        *,
        base: float = math.e,
        offset: float = 0.0,
        signed: bool = False,
        id: str | None = None,
    ):
        if base <= 0 or base == 1:
            raise ValueError(f"'base' must be positive and not 1, got {base!r}")
        if signed and offset != 0:
            raise ValueError(
                f"'offset' takes no part in the signed formula; got "
                f"offset={offset!r} with signed=True"
            )
        self.columns = columns
        self.base = base
        self.offset = offset
        self.signed = signed
        super().__init__(id, "log")
        # Populated by ``fit()``: the resolved column names, in frame order.
        self._estimates: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the selection and screen it for out-of-domain values."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_log(selected, offset=self.offset, signed=self.signed)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Log the columns resolved at fit time."""
        if self._estimates is None:
            raise RuntimeError("StepLog.transform() called before fit()")
        return apply_log(
            new_data,
            self._estimates,
            base=self.base,
            offset=self.offset,
            signed=self.signed,
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepLog.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, logged ones now float."""
        if self._schema_in is None or self._estimates is None:
            raise RuntimeError("StepLog.schema_out() called before fit()")
        return schema_out_for_touched(self._schema_in, set(self._estimates))

    def accepts_dtypes(self) -> AcceptedDtypes:
        """The logarithm is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``log(1) == 0`` moves zeros off zero; output stays dense."""
        return False
