"""Tests for :func:`shoez.apply_filter`.

Parametrized over pandas and polars to verify the backend-agnostic
contract: pandas in → pandas out, polars in → polars out.
"""

from typing import Any, Callable

import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez import apply_filter, fit_zero_variance

FrameFactory = Callable[[dict[str, list[Any]]], Any]

BACKENDS = [
    pytest.param(pd.DataFrame, id="pandas"),
    pytest.param(pl.DataFrame, id="polars"),
]


def _columns_of(df: Any) -> list[str]:
    """Backend-agnostic ``list[str]`` of column names.

    pandas exposes ``.columns`` as an ``Index``; polars exposes it as a
    plain list. ``list(df.columns)`` normalises both to the same shape
    for ``==`` comparison in assertions.
    """
    return list(df.columns)


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_drops_named_columns(make_frame: FrameFactory) -> None:
    """The named columns disappear; the unnamed ones stay."""
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    result = apply_filter(df, ["a", "c"])
    assert _columns_of(result) == ["b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_accepts_ndarray_from_fit(make_frame: FrameFactory) -> None:
    """The numpy array returned by ``fit_zero_variance`` plugs straight in."""
    df = make_frame({"a": [1, 1, 1], "b": [1, 2, 3]})
    drop = fit_zero_variance(df)
    assert isinstance(drop, np.ndarray)
    result = apply_filter(df, drop)
    assert _columns_of(result) == ["b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_accepts_plain_list(make_frame: FrameFactory) -> None:
    """A hand-written ``list[str]`` works (broad ``Iterable[str]`` typing)."""
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_filter(df, ["a"])
    assert _columns_of(result) == ["b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_accepts_tuple(make_frame: FrameFactory) -> None:
    """Tuples also work — confirms the ``Iterable[str]`` contract."""
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_filter(df, ("a",))
    assert _columns_of(result) == ["b"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_empty_vector_is_noop(make_frame: FrameFactory) -> None:
    """An empty ``drop_cols`` leaves the column set unchanged."""
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # Both an empty list and an empty numpy array should behave the same.
    assert _columns_of(apply_filter(df, [])) == ["a", "b"]
    assert _columns_of(apply_filter(df, np.array([], dtype=np.str_))) == [
        "a",
        "b",
    ]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_apply_silently_skips_unknown_columns(
    make_frame: FrameFactory,
) -> None:
    """Names not in ``data`` are ignored, not raised on.

    This is required for train/serve parity: if an earlier step already
    removed one of the columns, the filter must keep working.
    """
    df = make_frame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_filter(df, ["a", "nonexistent"])
    assert _columns_of(result) == ["b"]


def test_apply_preserves_backend_pandas() -> None:
    """pandas in → pandas out."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_filter(df, ["a"])
    assert isinstance(result, pd.DataFrame)


def test_apply_preserves_backend_polars() -> None:
    """polars in → polars out."""
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = apply_filter(df, ["a"])
    assert isinstance(result, pl.DataFrame)


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_round_trip_train_then_test(make_frame: FrameFactory) -> None:
    """Fit on training data, apply same vector to held-out data.

    This is the end-to-end train/serve parity check: even if the
    held-out frame is *missing* one of the to-drop columns (because
    something upstream already removed it), ``apply_filter`` mustn't
    explode — it should silently skip the absent name and drop the rest.
    """
    train = make_frame({"a": [1, 1, 1], "b": [1, 2, 3], "c": [7, 7, 7]})
    drop = fit_zero_variance(train)  # array(['a', 'c'])
    # "test" frame is missing 'c' — represents an upstream step having
    # already removed it.
    test = make_frame({"a": [1, 1, 1], "b": [10, 11, 12]})
    result = apply_filter(test, drop)
    assert _columns_of(result) == ["b"]
