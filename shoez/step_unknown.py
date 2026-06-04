"""Replace missing categorical levels with an explicit ``"unknown"`` level.

The *fit* half lives in :func:`fit_unknown` and the *apply* half in
:func:`apply_unknown`, both plain functions so the data-ops core can be
exercised on its own. :class:`StepUnknown` wraps them in the
:class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can fit
and apply it.

A ``ONE_TO_ONE`` step (see ``design/steps-catalog.md``): each selected
categorical column has its null values recoded to a single explicit level so
that "missing" becomes a value a model can learn from rather than a gap. Ports
R ``recipes``' ``step_unknown``.

Deviation from R (for now): recoded columns are returned as ``String`` rather
than as a categorical with an added level. Preserving the categorical dtype and
its level set is deferred until the schema subsystem lands.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
from shoez.types import AcceptedDtypes, ColSchema, Frame, Mode, SchemaInfo

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_unknown", "apply_unknown", "StepUnknown"]

#: Default level used to replace missing categorical values.
DEFAULT_UNKNOWN_LEVEL = "unknown"

#: Dtypes ``step_unknown`` is valid on — categorical-like columns.
_CATEGORICAL_DTYPES: set[type[nw.dtypes.DType]] = {nw.Categorical, nw.Enum, nw.String}


def fit_unknown(
    features: Frame,
    new_level: str = DEFAULT_UNKNOWN_LEVEL,
) -> list[str]:
    """Determine which columns to recode and check for level collisions.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all categorical-like. Passed in
        by the step's ``fit()`` method after selector resolution.
    new_level:
        The level that null values will be recoded to. Default ``"unknown"``.

    Returns
    -------
    list[str] — the column names to recode, in input order. Every selected
    column is returned (not only those with nulls in the training data), so that
    nulls appearing at transform time are still recoded — guaranteeing
    train/serve parity.

    A :class:`UserWarning` is emitted if ``new_level`` already appears as a value
    in any column, since recoding would then merge genuine observations into the
    "unknown" bucket.
    """
    collisions = [
        col
        for col in features.columns
        if bool((features[col].cast(nw.String) == new_level).any())
    ]
    if collisions:
        warnings.warn(
            f"The level {new_level!r} already exists in {len(collisions)} "
            f"column(s); recoded nulls will be merged with it: {collisions!r}.",
            stacklevel=2,
        )
    return list(features.columns)


def apply_unknown(
    features: Frame,
    columns: list[str],
    new_level: str = DEFAULT_UNKNOWN_LEVEL,
) -> Frame:
    """Recode null values in ``columns`` to ``new_level``.

    Parameters
    ----------
    features:
        A narwhals DataFrame to recode.
    columns:
        The column names to recode — the columns returned by :func:`fit_unknown`.
        Any column of ``features`` not in this list is passed through untouched.
    new_level:
        The level that null values are recoded to. Default ``"unknown"``.

    Returns
    -------
    narwhals.DataFrame — the input frame with each column in ``columns`` cast to
    ``String`` and its nulls filled with ``new_level``. Column names, order, and
    structure are unchanged.
    """
    exprs = [
        nw.col(col).cast(nw.String).fill_null(new_level).alias(col) for col in columns
    ]
    return features.with_columns(*exprs)


class StepUnknown(Step):
    """Recode missing categorical levels to an explicit ``"unknown"`` level.

    A ``ONE_TO_ONE`` step: each selected categorical column has its nulls
    replaced in place by ``new_level``. The work is done by :func:`fit_unknown`
    and :func:`apply_unknown`; this class wraps them in the
    :class:`~shoez.step.Step` lifecycle so a :class:`~shoez.recipe.Recipe` can
    fit and apply it.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: list[str],
        new_level: str = DEFAULT_UNKNOWN_LEVEL,
        id: str | None = None,
    ):
        self.columns = list(columns)
        self.new_level = new_level
        self.id = id if id is not None else "unknown"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        # Populated by ``fit()``.
        self._columns: list[str] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Resolve the columns to recode and check for level collisions."""
        features = _frame.select(training_data, self.columns)
        self._columns = fit_unknown(features, self.new_level)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=list(self.columns), dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Recode nulls in the fitted columns to ``new_level``. Never re-estimates."""
        if self._columns is None:
            raise RuntimeError("StepUnknown.transform() called before fit().")
        return apply_unknown(new_data, self._columns, self.new_level)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepUnknown.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, recoded ones now ``String``."""
        if self._schema_in is None or self._columns is None:
            raise RuntimeError("StepUnknown.schema_out() called before fit().")
        recoded = set(self._columns)
        dtypes = {
            col: ("String" if col in recoded else self._schema_in.dtypes[col])
            for col in self.columns
        }
        return ColSchema(columns=list(self.columns), dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Recoding missing levels is only meaningful on categorical columns."""
        return set(_CATEGORICAL_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_unknown`` never produces sparse output."""
        return False
