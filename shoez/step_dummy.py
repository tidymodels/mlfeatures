"""Dummy (treatment) encoding: expand discrete columns into k-1 indicators.

``fit_dummy`` decides which levels to retain (all except the first, after
sorting). ``apply_dummy`` performs the indicator expansion and is shared with
`shoez.step_one_hot`. ``StepDummy`` wraps both in the
`Step` lifecycle.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals.stable.v2 as nw

from shoez import _frame
from shoez.step import Step
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

__all__ = ["fit_dummy", "apply_dummy", "StepDummy", "DummyEstimates", "DISCRETE_DTYPES"]

DISCRETE_DTYPES: set[type[nw.dtypes.DType]] = {
    nw.String,
    nw.Boolean,
    nw.Categorical,
    nw.Enum,
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.Int128,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.UInt128,
}


@dataclass(frozen=True)
class DummyEstimates:
    """Per-column dummy-encoding state learned at fit time.

    Bundles the two things `apply_dummy()` needs so the encoding is pinned to
    the training set. Marked ``frozen=True`` so the fitted state can't be
    mutated and drift train-to-serve.

    Attributes
    ----------
    levels
        Retained levels per column (all except the reference level), as
        returned by `fit_dummy()`. Drives the indicator columns produced.
    all_known
        Full set of levels seen at fit time per column (retained *and*
        reference). Lets `apply_dummy()` distinguish the reference level
        from genuinely unseen levels at transform time.
    """

    levels: dict[str, list[Any]]
    all_known: dict[str, set[Any]]


def fit_dummy(selected: nw.DataFrame[Any]) -> dict[str, list[Any]]:
    """Identify the levels to retain for treatment (drop-one) dummy encoding.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all discrete (string, integer,
        boolean, categorical, or enum). Passed in by `StepDummy.fit()`
        after column selection.

    Returns
    -------
    dict[str, list[Any]]
        Maps each column name to its retained sorted levels — the full sorted
        level list minus the first element (the reference/dropped level).

    Raises
    ------
    ValueError
        If the frame has zero rows, or if any column has fewer than two unique
        non-null levels.
    """
    # Empty selections never reach here: the recipe detects a zero-column
    # selection and treats the step as the identity transform, so ``fit`` is
    # not called. See ``Recipe.fit`` / ``Step._empty_selection``.
    if selected.shape[0] == 0:
        raise ValueError(
            "cannot fit dummy encoding on a zero-row frame; no levels can be learned"
        )
    result: dict[str, list[Any]] = {}
    for col in selected.columns:
        levels: list[Any] = selected[col].drop_nulls().unique().sort().to_list()
        if len(levels) < 2:
            raise ValueError(
                f"column {col!r} has fewer than 2 unique non-null levels "
                f"({len(levels)} found); dummy encoding requires at least 2 levels"
            )
        result[col] = levels[1:]  # drop first element (reference level)
    return result


def apply_dummy(
    frame: Frame,
    levels: dict[str, list[Any]],
    col_name_fn: Callable[[str, str], str],
    keep_original_cols: bool,
    *,
    all_known: dict[str, set[Any]] | None = None,
) -> Frame:
    """Expand discrete columns into binary indicator columns.

    Shared transform utility used by both `StepDummy` and
    `StepOneHot`.

    Parameters
    ----------
    frame:
        Internal narwhals frame to transform.
    levels:
        Retained level mapping — the levels that produce indicator columns.
        For ``StepDummy`` this excludes the reference level; for
        ``StepOneHot`` it includes all levels.
    col_name_fn:
        Resolves ``(source_col, level_as_str)`` to the output column name.
    keep_original_cols:
        If ``False``, source columns are dropped after expansion.
    all_known:
        Full set of levels seen at fit time, keyed by column name. Used to
        detect truly unseen levels at transform time. When ``None``, falls back
        to ``levels`` (correct for ``StepOneHot`` where retained == all known).
        ``StepDummy`` passes the full level set (retained + reference) so that
        the reference level is not incorrectly flagged as unseen.
    """
    source_cols = list(levels.keys())

    for col, retained_levels in levels.items():
        if not retained_levels:
            continue

        out_names = [col_name_fn(col, str(lvl)) for lvl in retained_levels]

        # Detect truly unseen levels — values not seen at fit time at all.
        # The reference level (present in all_known but not in retained_levels)
        # is NOT unseen; it legitimately produces zeros in all indicator columns.
        known: set[Any] = (
            all_known[col] if all_known is not None else set(retained_levels)
        )
        col_values: list[Any] = frame[col].drop_nulls().unique().to_list()
        unseen: list[Any] = [v for v in col_values if v not in known]
        if unseen:
            warnings.warn(
                f"column {col!r} contains {len(unseen)} level(s) not seen "
                f"during fit: {unseen!r}; "
                "those rows will be null in all indicator columns",
                stacklevel=3,
            )

        # Indicator dtype. The ``when/then/otherwise`` chain below would otherwise
        # produce an ``Object`` column on the pandas backend (the ``lit(None)``
        # branch makes narwhals unify the branch types to object even when no row
        # is null), which is not numeric and so breaks downstream numeric steps
        # such as StepNormalize. Cast explicitly to fix this:
        #   * clean columns -> ``Int8`` (true binary indicators);
        #   * columns with null-source or unseen-level rows -> ``Float64``,
        #     because pandas' numpy ``Int8`` cannot hold the resulting NaN
        #     (``Float64`` is the only numeric dtype that preserves missing
        #     values across both pandas and polars).
        produces_missing = bool(unseen) or _frame.has_nulls(frame, col)
        indicator_dtype = nw.Float64 if produces_missing else nw.Int8

        # Build one indicator expression per retained level.
        # Null-source rows → null indicator.
        # Unseen-level rows → null indicator (treated the same as unknown).
        # Matching rows → 1; all other known rows → 0.
        exprs: list[Any] = []
        for lvl, out_name in zip(retained_levels, out_names):
            when_chain = (
                nw.when(nw.col(col) == lvl)
                .then(nw.lit(1))
                .when(nw.col(col).is_null())
                .then(nw.lit(None))
            )
            if unseen:
                when_chain = when_chain.when(nw.col(col).is_in(unseen)).then(
                    nw.lit(None)
                )
            exprs.append(
                when_chain.otherwise(nw.lit(0)).cast(indicator_dtype).alias(out_name)
            )
        frame = frame.with_columns(exprs)

    if not keep_original_cols:
        frame = _frame.drop(frame, source_cols)

    return frame


class StepDummy(Step):
    """Expand discrete columns into *k−1* binary indicator columns.

    Each selected column is replaced by *k−1* indicator columns where *k* is
    the number of unique non-null levels seen during ``fit()``. Indicators are
    ``nw.Int8`` (0/1); a column that contains null-source or unseen-level rows
    is ``nw.Float64`` instead (0.0/1.0/NaN), since pandas' ``Int8`` cannot hold
    the missing values. The first level in ascending sort order is the
    reference level and is dropped.

    Parameters
    ----------
    columns
        Columns to expand. Accepts a single column name, a list of names, a
        shoez selector such as ``Selectors.nominal_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.string()``, ``polars.selectors.string()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only discrete
        columns are meaningful here.
    name_fn
        Callable mapping ``(source_column, level_as_str)`` to the indicator
        column's name. Default ``None``, which names them ``"{column}_{level}"``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the indicators
        instead of being dropped. Default ``False``.
    id
        Stable step identifier. Default ``"dummy"``.

    Examples
    --------
    Each categorical column is replaced by *k−1* indicators; the first level in
    sort order (``blue``) is the dropped reference:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepDummy
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame({"color": ["red", "green", "blue", "red"]})
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDummy(s.nominal_features()))
    recipe
    ```

    `fit()` records the levels of each selected column:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the column with its indicators:

    ```{python}
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source column alongside them:

    ```{python}
    recipe = Recipe().add(StepDummy(s.nominal_features(), keep_original_cols=True))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
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
        self.id = id if id is not None else "dummy"
        self.trained = False
        self.output_role: str | None = "feature"
        self._estimates: DummyEstimates | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the sorted level list for each selected column."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        # Build the full known-level set (retained + reference) so that
        # apply_dummy does not flag the reference level as "unseen" at transform.
        self._estimates = DummyEstimates(
            levels=fit_dummy(selected),
            all_known={
                col: set(selected[col].drop_nulls().unique().to_list())
                for col in selected.columns
            },
        )
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        # Each indicator descends from exactly its source column (a1, a2 ← a).
        # The ordered list preserves any duplicate name (which the parents map,
        # keyed by name, would silently dedup).
        produced = [
            self._col_name(col, str(lvl))
            for col, lvls in self._estimates.levels.items()
            for lvl in lvls
        ]
        self._output_parents = {
            self._col_name(col, str(lvl)): [col]
            for col, lvls in self._estimates.levels.items()
            for lvl in lvls
        }
        self._schema_out = ColSchema(columns=produced)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Apply stored level lists to produce indicator columns."""
        if self._estimates is None:
            raise RuntimeError("StepDummy.transform() called before fit()")
        return apply_dummy(
            new_data,
            self._estimates.levels,
            self._col_name,
            self.keep_original_cols,
            all_known=self._estimates.all_known,
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepDummy.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the indicator columns this step produced. Known after ``fit()``."""
        if self._schema_out is None:
            raise RuntimeError("StepDummy.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each indicator column descends from its single source column."""
        if self._output_parents is None:
            raise RuntimeError("StepDummy.output_parents() called before fit()")
        return self._output_parents

    def produced_columns(self) -> list[str]:
        """Indicator names in order, preserving duplicates (see ``Step``)."""
        return list(self.schema_out().columns)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Accepts discrete columns only (see ``DISCRETE_DTYPES``)."""
        return set(DISCRETE_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._estimates is None:
            return
        for source_col, levels in self._estimates.levels.items():
            source_roles = roles.of(source_col)
            for role in source_roles:
                for lvl in levels:
                    roles.add(self._col_name(source_col, str(lvl)), role)
            if not self.keep_original_cols:
                for role in list(source_roles):
                    roles.remove(source_col, role)

    @property
    def produces_sparse(self) -> bool:
        """``StepDummy`` never produces sparse output."""
        return False

    def _col_name(self, col: str, level: str) -> str:
        if self.name_fn is not None:
            return self.name_fn(col, level)
        return f"{col}_{level}"
