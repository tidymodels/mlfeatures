"""One-hot encoding: expand discrete columns into k indicator columns.

``fit_one_hot`` identifies all unique levels per column (no reference level
dropped). Transform logic is shared with :mod:`shoez.step_dummy` via
:func:`~shoez.step_dummy.apply_dummy`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
from shoez.step_dummy import DISCRETE_DTYPES, apply_dummy
from shoez.types import (
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

__all__ = ["fit_one_hot", "StepOneHot"]


def fit_one_hot(features: nw.DataFrame[Any]) -> dict[str, list[Any]]:
    """Identify all levels for one-hot (full indicator) encoding.

    Parameters
    ----------
    features:
        A narwhals DataFrame whose columns are all discrete (string, integer,
        boolean, categorical, or enum). Passed in by :meth:`StepOneHot.fit`
        after column selection.

    Returns
    -------
    dict[str, list[Any]]
        Maps each column name to its full sorted level list (all *k* levels
        retained; no reference level is dropped).

    Raises
    ------
    ValueError
        If the frame has zero rows, or if any column has fewer than two unique
        non-null levels.
    """
    # Empty selection (no columns) is a no-op — there is nothing to encode.
    # Check this before the zero-row guard: a 0-column frame collapses to
    # shape (0, 0) in pandas, which would otherwise look like a zero-row frame.
    if len(features.columns) == 0:
        return {}
    if features.shape[0] == 0:
        raise ValueError(
            "Cannot fit one-hot encoding on a zero-row frame — "
            "no levels can be learned."
        )
    result: dict[str, list[Any]] = {}
    for col in features.columns:
        levels: list[Any] = features[col].drop_nulls().unique().sort().to_list()
        if len(levels) < 2:
            raise ValueError(
                f"Column {col!r} has fewer than 2 unique non-null levels "
                f"({len(levels)} found). One-hot encoding requires at least 2 levels."
            )
        result[col] = levels
    return result


class StepOneHot(Step):
    """Expand discrete columns into *k* binary indicator columns.

    A ``ONE_TO_MANY`` step: each selected column is replaced by *k* indicator
    columns where *k* is the number of unique non-null levels seen during
    ``fit()``. Indicators are ``nw.Int8`` (0/1); a column that contains
    null-source or unseen-level rows is ``nw.Float64`` instead (0.0/1.0/NaN),
    since pandas' ``Int8`` cannot hold the missing values. Unlike
    :class:`~shoez.step_dummy.StepDummy`, no reference level is dropped.
    Transform logic is shared via :func:`~shoez.step_dummy.apply_dummy`.

    Selectors are not yet implemented, so the columns to consider are passed
    explicitly as a list of names.
    """

    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: Selector,
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.name_fn = name_fn
        self.keep_original_cols = keep_original_cols
        self.id = id if id is not None else "one_hot"
        self.trained = False
        self.output_role: str | None = "predictor"
        self._levels: dict[str, list[Any]] | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the full sorted level list for each selected column."""
        features = _frame.select(training_data, self.columns, info.roles)
        self._levels = fit_one_hot(features)
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes)
        out_cols = [
            self._col_name(col, str(lvl))
            for col, lvls in self._levels.items()
            for lvl in lvls
        ]
        self._schema_out = ColSchema(columns=out_cols)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply stored level lists to produce indicator columns."""
        if self._levels is None:
            raise RuntimeError("StepOneHot.transform() called before fit().")
        return apply_dummy(
            new_data, self._levels, self._col_name, self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepOneHot.schema_in() called before fit().")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the indicator columns this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepOneHot.schema_out() called before fit().")
        return self._schema_out

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Accepts discrete columns; rejects float and decimal."""
        return set(DISCRETE_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._levels is None:
            return
        for source_col, levels in self._levels.items():
            source_roles = roles.of(source_col)
            for role in source_roles:
                for lvl in levels:
                    roles.add(self._col_name(source_col, str(lvl)), role)
            if not self.keep_original_cols:
                for role in list(source_roles):
                    roles.remove(source_col, role)

    @property
    def produces_sparse(self) -> bool:
        """``StepOneHot`` never produces sparse output."""
        return False

    def _col_name(self, col: str, level: str) -> str:
        if self.name_fn is not None:
            return self.name_fn(col, level)
        return f"{col}_{level}"
