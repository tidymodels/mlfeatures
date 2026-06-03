"""Zero-variance filter: find columns with a single unique value.

This is a precursor to ``step_zv`` (see ``design/steps-catalog.md``). It
exposes the *fit* half of the zero-variance rule as a plain function so
the logic can be exercised on its own, then later wrapped in a ``Step``
subclass without rewriting it.

The *apply* half (dropping the named columns from a dataframe) lives in
:mod:`shoez.apply_shared`, because every REMOVING-mode step
(``step_zv``, ``step_corr``, and so on) ends with the same operation.
"""

# ``from __future__ import annotations`` makes all type hints strings at
# runtime, which lets us use modern syntax (e.g. ``list[str]``) on older
# Python versions and removes any cost of evaluating the annotations.
from __future__ import annotations

import narwhals.stable.v1 as nw
import numpy as np
import numpy.typing as npt
from narwhals.typing import IntoFrame


def fit_zero_variance(features: IntoFrame) -> npt.NDArray[np.str_]:
    """Find columns with a single unique non-null value.

    A column is "zero-variance" if every observed (non-null) value is
    the same. Missing values are ignored, so a column of ``[5, 5, NaN]``
    counts as zero-variance because there is only one *observed* value.
    A column whose values are *all* missing has zero observed values,
    which is also treated as zero-variance (no information).

    Works for any dataframe narwhals supports (pandas, polars, etc.).

    Parameters
    ----------
    features
        Tabular input holding the candidate predictor columns. Anything
        narwhals can wrap with :func:`narwhals.from_native`. Most
        commonly a pandas or polars ``DataFrame``.

    Returns
    -------
    numpy.ndarray
        A 1-D array of column names (string dtype) that should be
        dropped. The Python analogue of an R character vector. Pass
        this array straight to :func:`shoez.apply_filter` to remove the
        offending columns from any frame with the same schema.

    Examples
    --------
    >>> import pandas as pd
    >>> from shoez import fit_zero_variance
    >>> df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    >>> fit_zero_variance(df).tolist()
    ['a']
    """
    # ``from_native`` wraps the input in a narwhals frame so the rest of
    # the function reads the same regardless of whether the user gave us
    # pandas or polars. ``eager_only=True`` rejects lazy frames — we
    # need to read values right now, not build a query plan.
    frame = nw.from_native(features, eager_only=True)

    # For each column, count distinct non-null values. ``<= 1`` is the
    # zero-variance rule:
    #   - 1 unique value  → constant column.
    #   - 0 unique values → every row is NA (no information either way).
    # ``frame[c]`` returns a narwhals ``Series``; ``.drop_nulls()``
    # strips NAs *before* counting, so a column of ``[5, 5, NaN]`` has
    # ``n_unique() == 1`` and is flagged.
    drop = [c for c in frame.columns if frame[c].drop_nulls().n_unique() <= 1]

    # ``np.asarray([])`` defaults to a ``float64`` array, so an empty
    # result would have a different dtype from a non-empty one. Forcing
    # ``dtype=np.str_`` keeps the return type stable in both cases.
    return np.asarray(drop, dtype=np.str_)
