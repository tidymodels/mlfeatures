"""Broad numeric-dtype coverage for the five numeric-accepting steps.

``step_normalize``, ``step_polynomial``, ``step_class_distance``,
``step_impute_mean``, and ``step_impute_median`` all declare acceptance of every
dtype in ``NUMERIC_DTYPES`` (``types.py``), but their per-step tests only
exercise ``Float64`` + ``Int64``. This module closes that gap by running each
step over its accepted numeric dtypes and asserting the expected output dtype.

Dtypes fall into two groups:

* **numpy-representable** (the 10 below) — the realistic scikit-learn boundary
  surface; these round-trip through numpy/pandas and every step accepts them.
* **polars-native** (``Int128``, ``UInt128``, ``Decimal``) — no numpy
  equivalent. ``step_normalize`` and the impute steps use polars-native ops and
  accept all three. ``step_polynomial`` and ``step_class_distance`` route
  through ``Series.to_numpy()``: ``Decimal`` survives (it converts cleanly) but
  ``Int128``/``UInt128`` *panic* in polars' numpy bridge. Those two dtypes are
  therefore excluded for the numpy-backed steps — a known overstatement in their
  ``accepts_dtypes()`` (all of ``NUMERIC_DTYPES``) versus what they can actually
  process at the numpy boundary.

Output-dtype contract asserted here: a **non-float** input (integer or decimal)
is promoted to ``Float64``; a **float** input stays float (``Float32`` is *not*
spuriously upcast to ``Float64``).

Input frames are built with polars regardless of group: polars carries nulls
inside integer columns natively (pandas integers cannot), which the impute
steps need to exercise the null→float promotion path.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import narwhals.stable.v2 as nw
import polars as pl
import pytest

from shoez import (
    Recipe,
    Step,
    StepClassDistance,
    StepImputeMean,
    StepImputeMedian,
    StepNormalize,
    StepPolynomial,
)

# The 10 dtypes that round-trip through numpy/pandas (the sklearn surface).
NUMPY_REPRESENTABLE = [
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
]

# polars-native numeric dtypes with no numpy equivalent. Decimal needs a scale.
DECIMAL = pl.Decimal(scale=0)
INT128_DTYPES = [pl.Int128, pl.UInt128]

# Steps backed by polars-native ops accept every numeric dtype.
ALL_NUMERIC = [*NUMPY_REPRESENTABLE, *INT128_DTYPES, DECIMAL]
# Steps that route through numpy accept the 10 + Decimal, but not Int128/UInt128.
NUMPY_BACKED = [*NUMPY_REPRESENTABLE, DECIMAL]


def _id(dtype: Any) -> str:
    return str(dtype)


def _num(dtype: Any, values: list[Any]) -> pl.Series:
    return pl.Series(values, dtype=dtype, strict=False)


# A scenario: a name, the dtypes it supports, an input-frame builder (numeric
# columns cast to ``dtype``), a step factory, and the output columns to check.
Scenario = tuple[
    str, list[Any], Callable[[Any], pl.DataFrame], Callable[[], Step], list[str]
]

SCENARIOS: list[Scenario] = [
    (
        "normalize",
        ALL_NUMERIC,
        lambda dt: pl.DataFrame({"x": _num(dt, [1, 2, 3, 4])}),
        lambda: StepNormalize(["x"]),
        ["x"],
    ),
    (
        "polynomial",
        NUMPY_BACKED,
        lambda dt: pl.DataFrame({"x": _num(dt, [1, 2, 3, 4])}),
        lambda: StepPolynomial(["x"]),
        ["x_1", "x_2"],
    ),
    (
        "class_distance",
        NUMPY_BACKED,
        lambda dt: pl.DataFrame(
            {
                "x": _num(dt, [1, 2, 3, 4]),
                "y": _num(dt, [2, 4, 6, 8]),
                "label": ["a", "a", "b", "b"],
            }
        ),
        lambda: StepClassDistance(["x", "y"], outcome="label"),
        ["distance_a", "distance_b"],
    ),
    # Impute scenarios include a null so the int→float promotion path runs.
    (
        "impute_mean",
        ALL_NUMERIC,
        lambda dt: pl.DataFrame({"x": _num(dt, [1, 2, None, 4])}),
        lambda: StepImputeMean(["x"]),
        ["x"],
    ),
    (
        "impute_median",
        ALL_NUMERIC,
        lambda dt: pl.DataFrame({"x": _num(dt, [1, 2, None, 4])}),
        lambda: StepImputeMedian(["x"]),
        ["x"],
    ),
]

# Flatten to (scenario, dtype) pairs so each combination is its own test case.
CASES = [
    pytest.param(
        name,
        build_input,
        build_step,
        check_cols,
        dtype,
        is_float,
        id=f"{name}-{_id(dtype)}",
    )
    for name, dtypes, build_input, build_step, check_cols in SCENARIOS
    for dtype in dtypes
    for is_float in [dtype in (pl.Float32, pl.Float64)]
]


@pytest.mark.parametrize(
    "name,build_input,build_step,check_cols,dtype,float_input", CASES
)
def test_numeric_dtype_output(
    name: str,
    build_input: Callable[[Any], pl.DataFrame],
    build_step: Callable[[], Step],
    check_cols: list[str],
    dtype: Any,
    float_input: bool,
) -> None:
    df = build_input(dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = Recipe(steps=[build_step()]).fit(df).transform(df)
    schema = nw.from_native(out).schema
    for col in check_cols:
        if float_input:
            # Float input stays float; Float32 must not be upcast to Float64.
            assert schema[col].is_float(), (
                f"{name}: column {col!r} should stay float for input {dtype}, "
                f"got {schema[col]}"
            )
        else:
            assert schema[col] == nw.Float64, (
                f"{name}: column {col!r} should promote to Float64 for input "
                f"{dtype}, got {schema[col]}"
            )
