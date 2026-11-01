"""Near-zero-variance drop: find columns that are sparse and unbalanced.

The *fit* half of the near-zero-variance rule lives in
`fit_drop_near_zero_variance()` as a plain function so the logic can be
exercised on its own; `StepDropNearZeroVariance` supplies it as the
`_drop_candidates()` rule for the fit/transform/schema lifecycle shared with
the other ``step_drop_*`` steps in `mlfeatures._drop.ColumnDropStep`.
"""

from __future__ import annotations

from mlfeatures._drop import ColumnDropStep
from mlfeatures.step_drop_zero_variance import fit_drop_zero_variance
from mlfeatures.types import ANY_DTYPE, AcceptedDtypes, Frame, Selector

__all__ = ["fit_drop_near_zero_variance", "StepDropNearZeroVariance"]


def fit_drop_near_zero_variance(
    selected: Frame, *, freq_cut: float, unique_cut: float
) -> list[str]:
    """Find columns that are zero-variance or near-zero-variance.

    A column is flagged if it is zero-variance (see `fit_drop_zero_variance()`,
    whose rule this delegates to rather than re-deriving), or if both of the
    following hold: the ratio of the most common non-null value's count to
    the second most common value's count exceeds ``freq_cut``, and the
    percentage of unique non-null values out of the total row count is at or
    below ``unique_cut``.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the candidate
        feature columns. Native frames are wrapped via
        `mlfeatures._frame.from_native()` at the recipe boundary.
    freq_cut
        Threshold for the frequency ratio; a column is only flagged as
        near-zero-variance if its ratio exceeds this value.
    unique_cut
        Threshold, in percent, for the share of unique non-null values; a
        column is only flagged as near-zero-variance if its percentage is at
        or below this value.

    Returns
    -------
    list[str]
        Column names to drop, in their input order. Pair with
        `mlfeatures._frame.apply_drop()` to remove the offending columns
        from any frame with the same schema.
    """
    zero_variance = set(fit_drop_zero_variance(selected))
    flagged: list[str] = []
    for col in selected.columns:
        if col in zero_variance:
            flagged.append(col)
            continue

        total = selected[col].len()
        values = selected[col].drop_nulls()
        n_unique = values.n_unique()
        count_name = f"{col}_count"
        counts_frame = values.value_counts(name=count_name)
        counts = sorted(counts_frame[count_name].to_list(), reverse=True)
        freq_ratio = counts[0] / counts[1]
        pct_unique = 100 * n_unique / total

        if freq_ratio > freq_cut and pct_unique <= unique_cut:
            flagged.append(col)

    return flagged


class StepDropNearZeroVariance(ColumnDropStep):
    """Remove columns that are sparse and unbalanced (near-zero variance).

    A selected column is dropped if it has a single unique value (or is
    entirely missing), or if it is both highly concentrated on one value and
    has few unique values relative to the number of samples: the ratio of the
    most common value's count to the second most common value's count exceeds
    ``freq_cut``, and the percentage of unique values is at or below
    ``unique_cut``. No new columns are produced.

    See [Zero Variance Filter](https://feaz-book.com/too-many-zv) in *Feature
    Engineering A to Z* for background on this technique.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to consider for removal. Accepts a single column name, a list
        of names, a mlfeatures selector such as ``Selectors.features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Any dtype is
        accepted.
    freq_cut
        Threshold for the frequency ratio of the most common value to the
        second most common value. Defaults to ``95 / 5``.
    unique_cut
        Threshold, in percent, for the share of unique values. Defaults to
        ``10``, and must be between ``0`` and ``100``.
    id
        Stable step identifier. Defaults to ``"nzv"``; a second step of this
        type in the same recipe becomes ``"nzv_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    ``sparse`` is a single value repeated 999 times with one outlier, so it is
    dropped; ``x`` is kept:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepDropNearZeroVariance
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {
            "x": list(range(1000)),
            "sparse": [1.0] + [0.0] * 999,
        }
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepDropNearZeroVariance(s.numeric_features()))
    recipe
    ```

    `fit()` computes each column's frequency ratio and percent of unique
    values:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` drops the sparse, unbalanced one:

    ```{python}
    fitted.transform(df)
    ```

    A tighter ``unique_cut`` can keep columns that would otherwise be dropped:

    ```{python}
    Recipe().add(
        StepDropNearZeroVariance(s.numeric_features(), unique_cut=0.05)
    ).fit(df).transform(df)
    ```
    """

    def __init__(
        self,
        columns: Selector,
        *,
        freq_cut: float = 95 / 5,
        unique_cut: float = 10,
        id: str | None = None,
    ):
        if not 0 <= unique_cut <= 100:
            raise ValueError(
                f"'unique_cut' must be between 0 and 100, got {unique_cut!r}"
            )
        if freq_cut < 0:
            raise ValueError(f"'freq_cut' must be non-negative, got {freq_cut!r}")
        self.freq_cut = freq_cut
        self.unique_cut = unique_cut
        super().__init__(columns, id=id, default_id="nzv")

    def _drop_candidates(self, selected: Frame) -> list[str]:
        return fit_drop_near_zero_variance(
            selected, freq_cut=self.freq_cut, unique_cut=self.unique_cut
        )

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Near-zero variance is meaningful for any dtype, not just numerics."""
        return ANY_DTYPE
