"""Tests for :func:`mlfeatures.step_drop_near_zero_variance.fit_drop_near_zero_variance`.

Every test is parametrized over a *backend* factory so the same logic
runs against both pandas and polars — that's the contract the narwhals
abstraction is supposed to give us. Native frames are wrapped via
:func:`mlfeatures._frame.from_native` before the fitter sees them, matching
the lifecycle (the recipe boundary does the wrapping; steps only ever
see a :data:`~mlfeatures.types.Frame`).
"""

from typing import Any, Callable

import pandas as pd
import polars as pl
import pytest

from mlfeatures import _frame as f
from mlfeatures.step_drop_near_zero_variance import fit_drop_near_zero_variance

FrameFactory = Callable[[dict[str, list[Any]]], Any]

BACKENDS = [
    pytest.param(pd.DataFrame, id="pandas"),
    pytest.param(pl.DataFrame, id="polars"),
]

DEFAULTS = {"freq_cut": 95 / 5, "unique_cut": 10.0}


def _wrap(make_frame: FrameFactory, data: dict[str, list[Any]]) -> f.Frame:
    """Build a native frame for the backend, then wrap it as a Frame."""
    return f.from_native(make_frame(data))


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_single_value_column(make_frame: FrameFactory) -> None:
    """A column whose every row is the same value is zero-variance."""
    frame = _wrap(make_frame, {"a": [1, 1, 1], "b": [1, 2, 3]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_all_na_column(make_frame: FrameFactory) -> None:
    """A column of all NA has zero observed values, so it is flagged."""
    frame = _wrap(make_frame, {"a": [None, None, None], "b": [1, 2, 3]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_nas_in_zero_variance_check(make_frame: FrameFactory) -> None:
    """A column of [5, 5, NA] is zero-variance (1 unique non-null)."""
    frame = _wrap(make_frame, {"a": [5, 5, None], "b": [1, 2, 3]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_sparse_unbalanced_column(make_frame: FrameFactory) -> None:
    """1 value repeated 999 times plus 1 outlier: high ratio, low pct unique."""
    frame = _wrap(make_frame, {"a": [1.0] * 999 + [0.0], "b": list(range(1000))})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_balanced_two_level_column(make_frame: FrameFactory) -> None:
    """A 50/50 split has freq_ratio == 1, well under the default cutoff."""
    frame = _wrap(make_frame, {"a": [1.0] * 10 + [0.0] * 10})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_ignores_high_ratio_but_many_unique_values(
    make_frame: FrameFactory,
) -> None:
    """High freq_ratio alone isn't enough; pct_unique must also clear unique_cut."""
    # 50 copies of 0.0, plus 49 other distinct values: pct_unique = 50/99 > 10.
    frame = _wrap(make_frame, {"a": [0.0] * 50 + [float(i) for i in range(1, 50)]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_boundary_freq_ratio_not_flagged(make_frame: FrameFactory) -> None:
    """freq_ratio exactly equal to freq_cut is not flagged (needs strictly >)."""
    # 19 copies of 1.0, 1 copy of 0.0: freq_ratio == 19 == default freq_cut.
    frame = _wrap(make_frame, {"a": [1.0] * 19 + [0.0]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_boundary_pct_unique_is_flagged(make_frame: FrameFactory) -> None:
    """pct_unique exactly equal to unique_cut is flagged (boundary is kept)."""
    # 20 copies of 1.0, 1 copy of 0.0: freq_ratio = 20 > 19; pct_unique = 200/21.
    frame = _wrap(make_frame, {"a": [1.0] * 20 + [0.0]})
    result = fit_drop_near_zero_variance(frame, freq_cut=19, unique_cut=200 / 21)
    assert result == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_pct_unique_denominator_includes_nulls(make_frame: FrameFactory) -> None:
    """pct_unique divides by the total row count, nulls included.

    20 copies of 1.0, 1 copy of 0.0, plus 100 nulls: freq_ratio (non-null
    only) is 20 > 19. Dividing by the total row count (121) gives pct_unique
    ~= 1.65, which clears ``unique_cut=5``; dividing by the non-null count
    (21) alone would give ~9.52, which would not. This distinguishes the two.
    """
    frame = _wrap(make_frame, {"a": [1.0] * 20 + [0.0] + [None] * 100})
    assert fit_drop_near_zero_variance(frame, freq_cut=19, unique_cut=5) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_empty_when_nothing_flagged(make_frame: FrameFactory) -> None:
    """If every column varies enough, the returned list is empty."""
    frame = _wrap(make_frame, {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == []


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_flags_multiple_columns_order_preserved(
    make_frame: FrameFactory,
) -> None:
    """Several flagged columns are all returned, in input order."""
    frame = _wrap(
        make_frame,
        {"a": [1, 1, 1], "b": [1, 2, 3], "c": [7, 7, 7]},
    )
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a", "c"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_handles_string_columns(make_frame: FrameFactory) -> None:
    """Near-zero-variance is type-agnostic; strings work like numbers."""
    frame = _wrap(
        make_frame, {"a": ["x"] * 999 + ["y"], "b": [str(i) for i in range(1000)]}
    )
    assert fit_drop_near_zero_variance(frame, **DEFAULTS) == ["a"]


@pytest.mark.parametrize("make_frame", BACKENDS)
def test_fit_returns_list(make_frame: FrameFactory) -> None:
    """The return type is a plain Python list."""
    frame = _wrap(make_frame, {"a": [1, 1, 1], "b": [1, 2, 3]})
    result = fit_drop_near_zero_variance(frame, **DEFAULTS)
    assert isinstance(result, list)
