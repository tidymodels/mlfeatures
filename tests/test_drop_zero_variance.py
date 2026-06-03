"""Tests for :func:`shoez.step_drop_zero_variance.fit_drop_zero_variance`.

Every test is parametrized over a *backend* factory so the same logic
runs against both pandas and polars — that's the contract the narwhals
abstraction is supposed to give us. Native frames are wrapped via
:func:`shoez._frame.from_native` before the fitter sees them, matching
the lifecycle (the recipe boundary does the wrapping; steps only ever
see a :data:`~shoez.types.Frame`).
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez import _frame as f
from shoez.step_drop_zero_variance import fit_drop_zero_variance

FrameFactory = Callable[[dict[str, list[Any]]], Any]

BACKENDS = [
    pytest.param(pd.DataFrame, id="pandas"),
    pytest.param(pl.DataFrame, id="polars"),
]


def _wrap(make_frame: FrameFactory, data: dict[str, list[Any]]) -> f.Frame:
    """Build a native frame for the backend, then wrap it as a Frame."""
    return f.from_native(make_frame(data))


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_single_value_column(make_frame: FrameFactory) -> None:
    """A column whose every row is the same value is zero-variance."""
    frame = _wrap(make_frame, {"a": [1, 1, 1], "b": [1, 2, 3]})
    assert fit_drop_zero_variance(frame).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_varying_column(make_frame: FrameFactory) -> None:
    """A column with >1 unique value is never flagged."""
    frame = _wrap(make_frame, {"a": [1, 2, 3]})
    assert fit_drop_zero_variance(frame).tolist() == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_empty_when_nothing_constant(
    make_frame: FrameFactory,
) -> None:
    """If every column varies, the returned vector is empty."""
    frame = _wrap(make_frame, {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert fit_drop_zero_variance(frame).tolist() == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_multiple_constant_columns(
    make_frame: FrameFactory,
) -> None:
    """Several constant columns are all returned (order preserved)."""
    frame = _wrap(make_frame, {"a": [1, 1, 1], "b": [1, 2, 3], "c": [7, 7, 7]})
    assert fit_drop_zero_variance(frame).tolist() == ["a", "c"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_nas(make_frame: FrameFactory) -> None:
    """A column of [5, 5, NA] is zero-variance (1 unique non-null)."""
    frame = _wrap(make_frame, {"a": [5, 5, None], "b": [1, 2, 3]})
    assert fit_drop_zero_variance(frame).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_all_na_column(make_frame: FrameFactory) -> None:
    """A column of all NA has zero observed values → flagged."""
    frame = _wrap(make_frame, {"a": [None, None, None], "b": [1, 2, 3]})
    assert fit_drop_zero_variance(frame).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_single_row(make_frame: FrameFactory) -> None:
    """A single-row frame: every column is trivially constant."""
    frame = _wrap(make_frame, {"a": [1], "b": [2], "c": [3]})
    assert fit_drop_zero_variance(frame).tolist() == ["a", "b", "c"]


def test_fit_handles_empty_frame_pandas() -> None:
    """A 0-row pandas frame: every column has 0 observed values → all flagged.

    Built separately from the parametrized cases because pandas and
    polars need different empty-frame construction (polars infers
    column dtypes from the data and so won't accept an empty list).
    """
    native = pd.DataFrame(
        {
            "a": pd.Series([], dtype="float64"),
            "b": pd.Series([], dtype="int64"),
        }
    )
    assert fit_drop_zero_variance(f.from_native(native)).tolist() == ["a", "b"]


def test_fit_handles_empty_frame_polars() -> None:
    """A 0-row polars frame: same — all columns flagged."""
    native = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Float64, "b": pl.Int64})
    assert fit_drop_zero_variance(f.from_native(native)).tolist() == ["a", "b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_handles_string_columns(make_frame: FrameFactory) -> None:
    """`step_zv` accepts ANY_DTYPE — strings work as well as numbers."""
    frame = _wrap(make_frame, {"a": ["x", "x", "x"], "b": ["x", "y", "z"]})
    assert fit_drop_zero_variance(frame).tolist() == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_ndarray_of_strings(make_frame: FrameFactory) -> None:
    """The return type is a numpy array, not a plain Python list."""
    frame = _wrap(make_frame, {"a": [1, 1, 1], "b": [1, 2, 3]})
    result = fit_drop_zero_variance(frame)
    assert isinstance(result, np.ndarray)
    # ``kind == "U"`` means "unicode string"; covers ``<U1``, ``<U7``,
    # etc., without locking the test to a specific string length.
    assert result.dtype.kind == "U"


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_string_dtype_even_when_empty(
    make_frame: FrameFactory,
) -> None:
    """The empty-result case still produces a string-typed array.

    The forced ``dtype=np.str_`` in :func:`fit_drop_zero_variance` exists for
    exactly this case — without it, ``np.asarray([])`` would default to
    ``float64`` and downstream code that indexes by name would explode.
    """
    frame = _wrap(make_frame, {"a": [1, 2, 3]})
    result = fit_drop_zero_variance(frame)
    assert result.dtype.kind == "U"
    assert result.size == 0
