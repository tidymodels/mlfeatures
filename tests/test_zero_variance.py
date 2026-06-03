"""Tests for :func:`shoez.fit_zero_variance`.

Every test is parametrized over a *backend* factory so the same logic
runs against both pandas and polars — that's the contract the narwhals
abstraction is supposed to give us.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez import fit_zero_variance

# A ``FrameFactory`` is a callable that turns a column-dict into a
# DataFrame of the chosen backend. Tests can build whichever shape they
# need via ``make_frame({"a": [...], ...})``.
FrameFactory = Callable[[dict[str, list[Any]]], Any]

# pytest will run every parametrized test once per entry in this list,
# tagging each run with the ``id=`` for readable test names like
# ``test_fit_flags_single_value_column[pandas]``.
BACKENDS = [
    pytest.param(pd.DataFrame, id="pandas"),
    pytest.param(pl.DataFrame, id="polars"),
]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_single_value_column(make_frame: FrameFactory) -> None:
    """A column whose every row is the same value is zero-variance."""
    df = make_frame({"a": [1, 1, 1], "b": [1, 2, 3]})
    assert fit_zero_variance(df).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_varying_column(make_frame: FrameFactory) -> None:
    """A column with >1 unique value is never flagged."""
    df = make_frame({"a": [1, 2, 3]})
    assert fit_zero_variance(df).tolist() == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_empty_when_nothing_constant(
    make_frame: FrameFactory,
) -> None:
    """If every column varies, the returned vector is empty."""
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert fit_zero_variance(df).tolist() == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_multiple_constant_columns(
    make_frame: FrameFactory,
) -> None:
    """Several constant columns are all returned (order preserved)."""
    df = make_frame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [7, 7, 7]})
    assert fit_zero_variance(df).tolist() == ["a", "c"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_nas(make_frame: FrameFactory) -> None:
    """A column of [5, 5, NA] is zero-variance (1 unique non-null)."""
    df = make_frame({"a": [5, 5, None], "b": [1, 2, 3]})
    assert fit_zero_variance(df).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_all_na_column(make_frame: FrameFactory) -> None:
    """A column of all NA has zero observed values → flagged."""
    df = make_frame({"a": [None, None, None], "b": [1, 2, 3]})
    assert fit_zero_variance(df).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_single_row(make_frame: FrameFactory) -> None:
    """A single-row frame: every column is trivially constant."""
    df = make_frame({"a": [1], "b": [2], "c": [3]})
    assert fit_zero_variance(df).tolist() == ["a", "b", "c"]


def test_fit_handles_empty_frame_pandas() -> None:
    """A 0-row pandas frame: every column has 0 observed values → all flagged.

    Built separately from the parametrized cases because pandas and
    polars need different empty-frame construction (polars infers
    column dtypes from the data and so won't accept an empty list).
    """
    df = pd.DataFrame(
        {
            "a": pd.Series([], dtype="float64"),
            "b": pd.Series([], dtype="int64"),
        }
    )
    assert fit_zero_variance(df).tolist() == ["a", "b"]


def test_fit_handles_empty_frame_polars() -> None:
    """A 0-row polars frame: same — all columns flagged."""
    df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Float64, "b": pl.Int64})
    assert fit_zero_variance(df).tolist() == ["a", "b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_handles_string_columns(make_frame: FrameFactory) -> None:
    """`step_zv` accepts ANY_DTYPE — strings work as well as numbers."""
    df = make_frame({"a": ["x", "x", "x"], "b": ["x", "y", "z"]})
    assert fit_zero_variance(df).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_ndarray_of_strings(make_frame: FrameFactory) -> None:
    """The return type is a numpy array, not a plain Python list."""
    df = make_frame({"a": [1, 1, 1], "b": [1, 2, 3]})
    result = fit_zero_variance(df)
    assert isinstance(result, np.ndarray)
    # ``kind == "U"`` means "unicode string"; covers ``<U1``, ``<U7``,
    # etc., without locking the test to a specific string length.
    assert result.dtype.kind == "U"


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_string_dtype_even_when_empty(
    make_frame: FrameFactory,
) -> None:
    """The empty-result case still produces a string-typed array.

    The forced ``dtype=np.str_`` in :func:`fit_zero_variance` exists for
    exactly this case — without it, ``np.asarray([])`` would default to
    ``float64`` and downstream code that indexes by name would explode.
    """
    df = make_frame({"a": [1, 2, 3]})
    result = fit_zero_variance(df)
    assert result.dtype.kind == "U"
    assert result.size == 0
