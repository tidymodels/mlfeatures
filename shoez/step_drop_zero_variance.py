"""Zero-variance drop: find columns with a single unique value.

This is a precursor to ``step_drop_zv`` (see ``design/steps-catalog.md``). It
exposes the *fit* half of the zero-variance rule as a plain function so
the logic can be exercised on its own, then later wrapped in a ``Step``
subclass without rewriting it.

The *apply* half (dropping the named columns) is :func:`shoez._frame.apply_drop`,
shared with every REMOVING-mode step (``step_drop_zv``, ``step_drop_corr``, ...).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from shoez.types import Frame


def fit_drop_zero_variance(features: Frame) -> npt.NDArray[np.str_]:
    """Find columns with a single unique non-null value.

    A column is "zero-variance" if every observed (non-null) value is the
    same. Missing values are ignored, so ``[5, 5, NaN]`` counts as
    zero-variance (one observed value). A column whose values are *all*
    missing has zero observed values and is also treated as zero-variance.

    Parameters
    ----------
    features
        Internal :data:`~shoez.types.Frame` (narwhals) holding the candidate
        predictor columns. Native frames are wrapped via
        :func:`shoez._frame.from_native` at the recipe boundary.

    Returns
    -------
    numpy.ndarray
        1-D string-dtype array of column names to drop — the Python
        analogue of an R character vector. Pair with
        :func:`shoez._frame.apply_drop` to remove the offending columns
        from any frame with the same schema.
    """
    drop = [c for c in features.columns if features[c].drop_nulls().n_unique() <= 1]
    # Forcing ``dtype=np.str_`` keeps the empty-result case string-typed;
    # otherwise ``np.asarray([])`` would default to float64.
    return np.asarray(drop, dtype=np.str_)
