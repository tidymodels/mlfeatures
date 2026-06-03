"""Tests for the internal narwhals frame boundary (:mod:`shoez._frame`).

Beyond exercising each boundary helper, this module carries the conformance
workload for issue #15: a prototype of ``step_normalize``'s data ops written once
against the frame API and run over **both** pandas and polars inputs. If narwhals
is thick enough to express normalize identically across backends — backend in,
same backend out, same values — the decision to use it as the full internal
representation holds. Gaps surfaced here feed ``design/`` notes.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import _frame as f

# Each backend is a (name, constructor) pair; the constructor turns a plain
# ``dict`` of columns into that backend's native frame. Tests parametrize over
# both to prove a single code path works identically on each.
Constructor = Callable[[dict[str, Any]], Any]
BACKENDS: list[tuple[str, Constructor]] = [
    ("pandas", lambda data: pd.DataFrame(data)),
    ("polars", lambda data: pl.DataFrame(data)),
]
BACKEND_IDS = [name for name, _ in BACKENDS]


@pytest.fixture(params=BACKENDS, ids=BACKEND_IDS)
def construct(request: pytest.FixtureRequest) -> Constructor:
    """A native-frame constructor for the backend under test."""
    return request.param[1]


# --- Boundary --------------------------------------------------------------


def test_from_native_then_to_native_round_trips(construct: Constructor) -> None:
    native = construct({"a": [1, 2, 3]})
    frame = f.from_native(native)
    out = f.to_native(frame)
    assert type(out) is type(native)


def test_columns(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1], "b": [2]}))
    assert f.columns(frame) == ["a", "b"]


# --- Column selection ------------------------------------------------------


def test_select_single_name_is_not_split_into_chars(construct: Constructor) -> None:
    # "ab" must select the one column "ab", not columns "a" and "b".
    frame = f.from_native(construct({"ab": [1], "c": [2]}))
    assert f.columns(f.select(frame, "ab")) == ["ab"]


def test_select_sequence_preserves_order(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1], "b": [2], "c": [3]}))
    assert f.columns(f.select(frame, ["c", "a"])) == ["c", "a"]


def test_drop(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1], "b": [2]}))
    assert f.columns(f.drop(frame, "a")) == ["b"]


# --- apply_drop (tolerant drop) -----------------------------------------


def test_apply_drop_drops_named_columns(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
    assert f.columns(f.apply_drop(frame, ["a", "c"])) == ["b"]


def test_apply_drop_empty_list_is_noop(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1, 2], "b": [3, 4]}))
    assert f.columns(f.apply_drop(frame, [])) == ["a", "b"]


def test_apply_drop_silently_skips_unknown_columns(construct: Constructor) -> None:
    """Names not in the frame are ignored, not raised on.

    Required for train/serve parity: if an earlier step already removed
    one of the columns, the drop must keep working.
    """
    frame = f.from_native(construct({"a": [1, 2], "b": [3, 4]}))
    assert f.columns(f.apply_drop(frame, ["a", "nonexistent"])) == ["b"]


# --- Type-aware ops --------------------------------------------------------


def test_is_numeric_and_numeric_columns(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1, 2], "b": ["x", "y"]}))
    assert f.is_numeric(frame, "a")
    assert not f.is_numeric(frame, "b")
    assert f.numeric_columns(frame) == ["a"]


def test_dtypes_keys_match_columns(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1], "b": ["x"]}))
    assert list(f.dtypes(frame)) == ["a", "b"]


# --- Categorical handling --------------------------------------------------


def test_pandas_categorical_ordered_and_unordered_detected() -> None:
    native = pd.DataFrame(
        {
            "ordered": pd.Categorical(["a", "b", "a"], ordered=True),
            "unordered": pd.Categorical(["a", "b", "a"], ordered=False),
            "plain": ["a", "b", "c"],
        }
    )
    frame = f.from_native(native)
    assert f.is_categorical(frame, "ordered")
    assert f.is_categorical(frame, "unordered")
    assert not f.is_categorical(frame, "plain")
    assert f.categorical_columns(frame) == ["ordered", "unordered"]


def test_polars_categorical_and_enum_detected() -> None:
    native = pl.DataFrame(
        {
            "cat": pl.Series(["a", "b", "a"], dtype=pl.Categorical),
            "enum": pl.Series(["a", "b", "a"], dtype=pl.Enum(["a", "b"])),
            "plain": ["a", "b", "c"],
        }
    )
    frame = f.from_native(native)
    assert f.is_categorical(frame, "cat")
    assert f.is_categorical(frame, "enum")
    assert not f.is_categorical(frame, "plain")
    assert f.categorical_columns(frame) == ["cat", "enum"]


# --- Missing-value awareness -----------------------------------------------


def test_null_counts_and_has_nulls(construct: Constructor) -> None:
    frame = f.from_native(construct({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]}))
    assert f.null_counts(frame) == {"a": 1, "b": 0}
    assert f.has_nulls(frame, "a")
    assert not f.has_nulls(frame, "b")


# --- step_normalize data-ops prototype -------------------------------------
#
# Not a real Step (no selectors / id / schema lineage — those land in later step
# issues). This is the data-ops core of step_normalize, written once against the
# frame API to validate narwhals across backends.


def _fit_normalize(
    frame: f.Frame, columns: list[str]
) -> dict[str, tuple[float, float]]:
    """Estimate per-column (mean, std) over the named numeric columns."""
    stats: dict[str, tuple[float, float]] = {}
    for col in columns:
        series = frame[col]
        stats[col] = (float(series.mean()), float(series.std()))
    return stats


def _transform_normalize(
    frame: f.Frame, stats: dict[str, tuple[float, float]]
) -> f.Frame:
    """Apply ``(x - mean) / std`` to each fitted column."""
    exprs = [
        ((nw.col(col) - mean) / std).alias(col) for col, (mean, std) in stats.items()
    ]
    return frame.with_columns(*exprs)


def test_step_normalize_prototype(construct: Constructor) -> None:
    train = f.from_native(construct({"a": [0.0, 2.0, 4.0], "g": ["x", "y", "z"]}))

    cols = f.numeric_columns(train)
    assert cols == ["a"]

    stats = _fit_normalize(train, cols)
    mean, std = stats["a"]
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(2.0)  # ddof=1 across both backends

    out = _transform_normalize(train, stats)

    # Backend preserved: pandas in -> pandas out, polars in -> polars out.
    assert type(f.to_native(out)) is type(f.to_native(train))

    # Values identical across backends.
    normalized = out["a"].to_list()
    assert normalized == pytest.approx([-1.0, 0.0, 1.0])

    # Non-numeric column passed through untouched.
    assert out["g"].to_list() == ["x", "y", "z"]


def test_step_normalize_prototype_transform_uses_train_stats(
    construct: Constructor,
) -> None:
    # transform() applies fitted stats to *new* data — it never re-estimates.
    train = f.from_native(construct({"a": [0.0, 2.0, 4.0]}))
    stats = _fit_normalize(train, ["a"])

    new = f.from_native(construct({"a": [2.0, 6.0]}))
    out = _transform_normalize(new, stats)

    # (2-2)/2 = 0 ; (6-2)/2 = 2  — using TRAIN mean/std, not new-data stats.
    assert out["a"].to_list() == pytest.approx([0.0, 2.0])


def test_step_normalize_prototype_matches_across_backends() -> None:
    data = {"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]}
    results = {}
    for name, construct in BACKENDS:
        frame = f.from_native(construct(data))
        stats = _fit_normalize(frame, f.numeric_columns(frame))
        out = _transform_normalize(frame, stats)
        results[name] = {c: out[c].to_list() for c in f.columns(out)}

    for col in ("a", "b"):
        assert results["pandas"][col] == pytest.approx(results["polars"][col])
        assert not any(math.isnan(v) for v in results["pandas"][col])
