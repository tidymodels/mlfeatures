"""Collapse infrequent categorical levels into a single ``"other"`` level.

The *fit* half lives in `fit_other()` and the *apply* half in `apply_other()`,
both plain functions so the data-ops core can be exercised on its own.
`StepOther` wraps them in the `Step` lifecycle so a `Recipe` can fit and apply
it.

Each selected categorical column has its rare levels pooled into one level, so a
long tail of levels a model cannot learn from becomes a single informative
bucket. Ports R ``recipes``' ``step_other``.

Deviation from R: where ``recipes`` gives ``step_other`` a single ``threshold``
argument whose units flip from a proportion to a count at 1.0, mlfeatures splits it
into ``threshold`` (a proportion) and ``min_count`` (a count) and refuses both at
once. The no-argument default still matches ``recipes``' ``threshold = 0.05``.

Deviation from R (for now): recoded columns are returned as ``String`` rather
than as a categorical with a rebuilt level set. Preserving the categorical dtype
is deferred until the schema subsystem lands, matching ``step_unknown``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals.stable.v2 as nw

from mlfeatures import _frame
from mlfeatures._level_recode import warn_level_collision
from mlfeatures.step import Step
from mlfeatures.types import (
    _CATEGORICAL_DTYPES,  # pyright: ignore[reportPrivateUsage]
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["fit_other", "apply_other", "StepOther"]

#: Default level that infrequent levels are pooled into.
DEFAULT_OTHER_LEVEL = "other"

#: Proportion used when neither ``threshold`` nor ``min_count`` is given.
DEFAULT_THRESHOLD = 0.05


def fit_other(
    selected: Frame,
    *,
    threshold: float | None = None,
    min_count: int | None = None,
    other_level: str = DEFAULT_OTHER_LEVEL,
) -> dict[str, list[str]]:
    """Determine, per column, which levels are frequent enough to keep.

    Parameters
    ----------
    selected:
        A narwhals DataFrame whose columns are all categorical-like. Passed in
        by the step's ``fit()`` method after selector resolution.
    threshold:
        Minimum proportion of the column's non-null rows a level must reach to
        be kept. Mutually exclusive with ``min_count`` — `StepOther` rejects
        both at once in its constructor — and when both are ``None`` this falls
        back to ``DEFAULT_THRESHOLD``, so calling the helper with no criterion
        behaves like the step's no-argument default.
    min_count:
        Minimum number of rows a level must appear in to be kept.
    other_level:
        The level infrequent levels will be pooled into. Used here only for the
        collision check. Default ``"other"``.

    Returns
    -------
    dict[str, list[str]] — a mapping from every column of ``selected`` to its
    kept levels, sorted so the estimate is deterministic and JSON-serializable.
    Every column is present (not only those with something to collapse), because
    the kept set is also what decides the fate of levels *unseen* at fit time:
    `apply_other()` pools those into ``other_level`` too, which is what keeps
    train and serve in step. Nulls are excluded from the counts and from the
    proportion's denominator; recoding them is ``step_unknown``'s job.

    A `UserWarning` is emitted if ``other_level`` already appears as a value in
    any column, since pooling would then merge genuine samples into the "other"
    bucket.
    """
    collisions: list[str] = []
    kept: dict[str, list[str]] = {}

    for col in selected.columns:
        # Drop nulls *before* casting: older polars (resolved on Python 3.10)
        # casts null to the literal string "None", which would then survive
        # drop_nulls and be counted as a level of its own.
        values = selected[col].drop_nulls().cast(nw.String)
        n_non_null = len(values)
        # Count in the backend rather than over a materialized Python list: the
        # level counts are the only part of fitting that scales with row count.
        # ``count_name`` cannot collide with ``col``, which carries the levels.
        count_name = f"{col}_count"
        counts_frame = values.value_counts(name=count_name)
        levels: list[str] = counts_frame[col].to_list()
        ns: list[int] = counts_frame[count_name].to_list()
        if other_level in levels:
            collisions.append(col)
        # At most one of the two criteria is set; a level on the boundary is
        # kept, matching R ``recipes``.
        if min_count is not None:
            keep = [level for level, n in zip(levels, ns) if n >= min_count]
        else:
            cutoff = DEFAULT_THRESHOLD if threshold is None else threshold
            keep = (
                [level for level, n in zip(levels, ns) if n / n_non_null >= cutoff]
                if n_non_null
                else []
            )
        kept[col] = sorted(keep)

    warn_level_collision(other_level, collisions, merges="pooled levels")
    return kept


def apply_other(
    selected: Frame,
    kept: dict[str, list[str]],
    other_level: str = DEFAULT_OTHER_LEVEL,
) -> Frame:
    """Pool every level outside ``kept`` into ``other_level``.

    Parameters
    ----------
    selected:
        A narwhals DataFrame to recode.
    kept:
        The ``{column: kept levels}`` mapping returned by `fit_other()`. Any
        column of ``selected`` not present as a key is passed through untouched.
    other_level:
        The level infrequent and unseen levels are pooled into. Default
        ``"other"``.

    Returns
    -------
    narwhals.DataFrame — the input frame with each mapped column cast to
    ``String`` and every value outside its kept set replaced by ``other_level``.
    Nulls stay null. Column names, order, and structure are unchanged.
    """
    # Keep the null rows out of the pooling entirely, with an outer ``is_null``
    # mask whose false branch is left implicit. The obvious spelling -- a
    # ``then(lit(None, String))`` branch -- does not survive older polars
    # (resolved on Python 3.10), which materializes that literal as the string
    # "None"; an omitted ``otherwise`` yields a real null on every backend.
    exprs = [
        nw.when(~nw.col(col).is_null())
        .then(
            nw.when(nw.col(col).cast(nw.String).is_in(levels))
            .then(nw.col(col).cast(nw.String))
            .otherwise(nw.lit(other_level))
        )
        .alias(col)
        for col, levels in kept.items()
    ]
    return selected.with_columns(*exprs)


class StepOther(Step):
    """Collapse infrequent categorical levels into a single ``"other"`` level.

    Each selected categorical column keeps the levels that are frequent enough
    in the training data and has every other level replaced in place by
    ``other_level``. Levels that never appeared during fitting are pooled into
    ``other_level`` as well, so new data can never introduce a level the rest of
    the recipe has not seen. Nulls stay null; use ``StepUnknown`` first when
    missingness should become a level of its own.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to collapse. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.nominal_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.string()``, ``polars.selectors.string()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only categorical
        columns are meaningful here.
    threshold
        Smallest share of a column's non-null rows a level may hold and still be
        kept, as a proportion strictly between 0 and 1. Default ``None``;
        mutually exclusive with ``min_count``, and when neither is given a
        proportion of ``0.05`` is used.
    min_count
        Smallest number of rows a level may appear in and still be kept, as a
        whole number of 1 or more. Default ``None``; mutually exclusive with
        ``threshold``.
    other_level
        The level that pooled levels are replaced with. Default ``"other"``. A
        `UserWarning` is emitted if a selected column already contains this
        level, since the pooled rows then become indistinguishable from real
        ones.
    id
        Stable step identifier. Defaults to ``"other"``; a second step of this
        type in the same recipe becomes ``"other_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    The rare levels are pooled in place into a single ``"other"`` level:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepOther
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {"city": ["NYC"] * 5 + ["LA"] * 4 + ["Reno", "Provo"]}
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepOther(s.nominal_features(), threshold=0.2))
    recipe
    ```

    `fit()` records which levels are frequent enough to keep:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` pools everything else:

    ```{python}
    fitted.transform(df)
    ```

    ``min_count`` states the cutoff as a number of rows instead of a share:

    ```{python}
    recipe = Recipe().add(StepOther(s.nominal_features(), min_count=5))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``other_level`` names the pooled level:

    ```{python}
    recipe = Recipe().add(
        StepOther(s.nominal_features(), threshold=0.2, other_level="rare")
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.ONE_TO_ONE

    def __init__(
        self,
        columns: Selector,
        *,
        threshold: float | None = None,
        min_count: int | None = None,
        other_level: str = DEFAULT_OTHER_LEVEL,
        id: str | None = None,
    ):
        if threshold is not None and min_count is not None:
            raise ValueError(
                "'threshold' and 'min_count' are mutually exclusive; set only one"
            )
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError(
                f"'threshold' must be a proportion strictly between 0 and 1, got "
                f"{threshold!r}; pass 'min_count' to give a count of rows instead"
            )
        if min_count is not None and (min_count < 1 or min_count != int(min_count)):
            # A fractional count would otherwise be accepted and silently act as
            # its ceiling, which is exactly the units ambiguity splitting
            # 'threshold' and 'min_count' set out to remove.
            raise ValueError(
                f"'min_count' must be a whole number of rows, 1 or greater, got "
                f"{min_count!r}"
            )
        self.columns = columns
        self.threshold = threshold
        self.min_count = min_count
        self.other_level = other_level
        super().__init__(id, "other")
        # Populated by ``fit()``: {column: kept levels} over the selection.
        self._estimates: dict[str, list[str]] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Record the levels frequent enough to keep in each selected column."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_other(
            selected,
            threshold=self.threshold,
            min_count=self.min_count,
            other_level=self.other_level,
        )
        dtypes = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes)
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Pool levels outside the stored kept sets. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepOther.transform() called before fit()")
        return apply_other(new_data, self._estimates, self.other_level)

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepOther.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the produced columns: same names, all recoded to ``String``."""
        if self._schema_in is None:
            raise RuntimeError("StepOther.schema_out() called before fit()")
        dtypes = dict.fromkeys(self._schema_in.columns, "String")
        return ColSchema(columns=self._schema_in.columns, dtypes=dtypes)

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Pooling levels is only meaningful on categorical columns."""
        return set(_CATEGORICAL_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        """``step_other`` never produces sparse output."""
        return False
