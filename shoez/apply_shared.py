"""Shared helpers used by multiple steps.

Right now there is just :func:`apply_filter`, the column-dropping half
of every REMOVING-mode step (``step_zv``, ``step_corr``, and so on). It
lives here so each fitter only has to compute *which* columns to drop.
The drop itself is one function, tested once, that every filter shares.
"""

from __future__ import annotations

# ``Iterable`` is the broadest typing-protocol for "a thing you can
# loop over". Using it (instead of ``list[str]``) lets callers pass the
# numpy array returned by ``fit_zero_variance``, a plain list, a tuple, or
# even a generator — anything iterable of strings is fine.
from typing import Iterable

import narwhals.stable.v1 as nw
from narwhals.typing import IntoFrameT


def apply_filter(data: IntoFrameT, drop_cols: Iterable[str]) -> IntoFrameT:
    """Drop the listed columns from a dataframe.

    The companion to fitter functions such as :func:`shoez.fit_zero_variance`.
    Each fitter inspects training data and returns *which* columns to
    remove; :func:`apply_filter` performs the removal. Splitting the work
    that way means the same set of names can be applied to both the
    training frame and any held-out (or new) frame, which guarantees
    train/serve parity (the same columns disappear at fit and at
    transform time).

    Names listed in ``drop_cols`` that aren't present in ``data`` are
    silently skipped. This mirrors the framework's policy of tolerating
    extra and missing schema columns at transform time (see
    ``design/lifecycle.md``) and is what makes the train/serve guarantee
    work when an upstream step has already dropped one of the columns.

    Parameters
    ----------
    data
        Tabular input. Anything narwhals can wrap. Most commonly a
        pandas or polars ``DataFrame``. The native type is preserved on
        output: pandas in produces pandas out, polars in produces polars
        out.
    drop_cols
        Names of columns to remove. Typically the output of a fitter
        such as :func:`shoez.fit_zero_variance`; a plain ``list`` or tuple
        of strings works equally well.

    Returns
    -------
    DataFrame
        A new dataframe of the same backend type as ``data`` with the
        listed columns removed. Columns not in ``data`` are ignored;
        no error is raised.

    Examples
    --------
    >>> import pandas as pd
    >>> from shoez import fit_zero_variance, apply_filter
    >>> df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    >>> drop = fit_zero_variance(df)
    >>> apply_filter(df, drop).columns.tolist()
    ['b']
    """
    # Wrap the user's native frame so the body can be written once,
    # regardless of backend.
    frame = nw.from_native(data, eager_only=True)

    # Filter ``drop_cols`` to names that actually exist in ``data``.
    # Without this, narwhals raises if asked to drop a missing column —
    # but for train/serve parity we want to no-op on absent names.
    present = [c for c in drop_cols if c in frame.columns]

    # ``DataFrame.drop`` in narwhals takes column names as *positional*
    # arguments, so we star-unpack the list (``*present`` becomes
    # ``"a", "c"``). ``to_native`` then unwraps the result back to the
    # user's original DataFrame type. The ``IntoFrameT`` generic in the
    # return annotation tells type-checkers "same type out as in".
    return nw.to_native(frame.drop(*present))
