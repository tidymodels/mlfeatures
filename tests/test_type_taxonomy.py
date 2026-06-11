"""Round-trip coverage for the dtype taxonomy (issue #16).

The taxonomy in ``design/schema.md`` (numeric/float/int, bool, str, categorical
ordered/unordered, date, datetime, duration, list) is realized as **narwhals
dtype classes** rather than a custom enum — see ``shoez/types.py`` and the
predicates in ``shoez/_frame.py``. This module proves narwhals round-trips every
category correctly from **both** polars and pandas: each predicate fires for its
own category and only its own, and ``dtype_category`` names it.

Backend coverage gaps are handled per category rather than skipped silently:

* **list** and **date** — pandas surfaces neither as a narwhals dtype: a
  variable-length list column has no first-class pandas dtype, and a column of
  ``datetime.date`` objects lands as ``object`` (narwhals ``Object``), not
  ``Date``. Both are exercised on polars only. Documented as pandas limitations,
  not defects. (``datetime`` *does* round-trip on pandas via ``datetime64``.)
* **categorical ordered/unordered** — narwhals normalizes polars ``Enum`` and
  pandas ``Categorical(ordered=True)`` to ``Enum`` (ordered), and polars
  ``Categorical`` / pandas ``Categorical(ordered=False)`` to ``Categorical``
  (unordered). Both backends express both, so both are tested.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Callable

import pandas as pd
import polars as pl
import pytest

from shoez import _frame as f

Constructor = Callable[[dict[str, Any]], Any]

# Categories every backend can express, as (column-name, builder-per-backend).
# Each builder takes a backend tag ("pandas"/"polars") and returns column values
# typed to the target category. The expected ``dtype_category`` label and the
# predicate that must fire are attached per case.


def _pl_cat(ordered: bool) -> pl.Series:
    values = ["a", "b", "a"]
    if ordered:
        return pl.Series(values, dtype=pl.Enum(["a", "b"]))
    return pl.Series(values, dtype=pl.Categorical)


def _pd_cat(ordered: bool) -> pd.Categorical:
    return pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=ordered)


# A shared frame with one column per cross-backend category.
def _common_columns(backend: str) -> dict[str, Any]:
    cat_ordered = _pl_cat(True) if backend == "polars" else _pd_cat(True)
    cat_unordered = _pl_cat(False) if backend == "polars" else _pd_cat(False)
    return {
        "f": [1.0, 2.0, 3.0],
        "i": pl.Series([1, 2, 3], dtype=pl.Int64)
        if backend == "polars"
        else pd.array([1, 2, 3], dtype="int64"),
        "b": [True, False, True],
        "s": ["a", "b", "c"],
        "cat_ordered": cat_ordered,
        "cat_unordered": cat_unordered,
        "dt": [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)],
        "dur": [timedelta(days=1), timedelta(days=2), timedelta(days=3)],
    }


BACKENDS: list[tuple[str, Constructor]] = [
    ("pandas", lambda data: pd.DataFrame(data)),
    ("polars", lambda data: pl.DataFrame(data)),
]
BACKEND_IDS = [name for name, _ in BACKENDS]

# (column, expected category, predicate) for the cross-backend columns.
CASES: list[tuple[str, str, Callable[[Any, str], bool]]] = [
    ("f", "float", f.is_double),
    ("i", "int", f.is_integer),
    ("b", "bool", f.is_boolean),
    ("s", "str", f.is_string),
    ("cat_ordered", "categorical_ordered", f.is_ordered_categorical),
    ("cat_unordered", "categorical_unordered", f.is_categorical),
    ("dt", "datetime", f.is_datetime),
    ("dur", "duration", f.is_duration),
]


@pytest.fixture(params=BACKENDS, ids=BACKEND_IDS)
def backend(request: pytest.FixtureRequest) -> str:
    name: str = request.param[0]
    return name


@pytest.fixture
def common_frame(backend: str) -> Any:
    construct = dict(BACKENDS)[backend]
    return f.from_native(construct(_common_columns(backend)))


@pytest.mark.parametrize("column,category,predicate", CASES, ids=[c[0] for c in CASES])
def test_dtype_category_round_trips(
    common_frame: Any,
    column: str,
    category: str,
    predicate: Callable[[Any, str], bool],
) -> None:
    """Each column classifies to its taxonomy category on both backends."""
    dtype = f.dtypes(common_frame)[column]
    assert f.dtype_category(dtype) == category
    assert predicate(common_frame, column)


@pytest.mark.parametrize(
    "column,predicate",
    [(c[0], c[2]) for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_category_predicates_are_exclusive(
    common_frame: Any,
    column: str,
    predicate: Callable[[Any, str], bool],
) -> None:
    """A predicate fires for its own column and no other column."""
    for other, _, _ in CASES:
        if other == column:
            continue
        # is_categorical is the umbrella predicate (ordered + unordered), so the
        # ordered column is legitimately categorical too; exclude that one pair.
        if predicate is f.is_categorical and other == "cat_ordered":
            continue
        assert not predicate(common_frame, other), (
            f"{predicate.__name__} should not fire for {other!r}"
        )


def test_ordered_vs_unordered_categorical(common_frame: Any) -> None:
    """Ordered (Enum) and unordered (Categorical) are distinguished on both backends."""
    assert f.is_ordered_categorical(common_frame, "cat_ordered")
    assert not f.is_ordered_categorical(common_frame, "cat_unordered")
    # Both are categorical under the umbrella predicate.
    assert f.is_categorical(common_frame, "cat_ordered")
    assert f.is_categorical(common_frame, "cat_unordered")


def test_list_dtype_polars_only() -> None:
    """List columns round-trip on polars; pandas has no narwhals-surfaced List dtype."""
    frame = f.from_native(
        pl.DataFrame({"lst": pl.Series([[1, 2], [3], [4, 5]], dtype=pl.List(pl.Int64))})
    )
    dtype = f.dtypes(frame)["lst"]
    assert f.dtype_category(dtype) == "list"
    assert f.is_list(frame, "lst")


def test_date_dtype_polars_only() -> None:
    """Date round-trips on polars; pandas stores datetime.date as object, not Date."""
    frame = f.from_native(
        pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]})
    )
    dtype = f.dtypes(frame)["date"]
    assert f.dtype_category(dtype) == "date"
    assert f.is_date(frame, "date")


def test_decimal_classifies_as_decimal() -> None:
    """Decimal is numeric but neither float nor int, so it gets its own category."""
    frame = f.from_native(
        pl.DataFrame({"d": pl.Series([1, 2, 3], dtype=pl.Decimal(scale=0))})
    )
    dtype = f.dtypes(frame)["d"]
    assert f.dtype_category(dtype) == "decimal"
    assert not f.is_double(frame, "d")
    assert not f.is_integer(frame, "d")
