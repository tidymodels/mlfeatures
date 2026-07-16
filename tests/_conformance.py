"""Shared helpers for the per-step conformance/infrastructure suite.

Kept in an underscore-prefixed module so pytest does not collect it as a test
file; the ``test_step_x_infrastructure.py`` files import from here to avoid
duplicating cross-cutting conformance logic.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import _frame
from shoez.types import DataFrame, SchemaInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shoez import Recipe
    from shoez.step import Step


def _is_missing(value: Any) -> bool:
    """Treat polars ``None`` and pandas ``NaN`` as the same missing marker."""
    if value is None:
        return True
    return isinstance(value, float) and math.isnan(value)


def _assert_columns_equal(
    name: str, pd_values: list[Any], pl_values: list[Any]
) -> None:
    """Assert one column's values match across backends, missing-aware."""
    assert len(pd_values) == len(pl_values), (
        f"column {name!r} length divergence: pandas {len(pd_values)} != "
        f"polars {len(pl_values)}"
    )
    for i, (pd_v, pl_v) in enumerate(zip(pd_values, pl_values)):
        pd_missing, pl_missing = _is_missing(pd_v), _is_missing(pl_v)
        if pd_missing or pl_missing:
            assert pd_missing and pl_missing, (
                f"column {name!r} row {i} missing-value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )
            continue
        if isinstance(pd_v, float) or isinstance(pl_v, float):
            assert pd_v == pytest.approx(pl_v), (
                f"column {name!r} row {i} value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )
        else:
            assert pd_v == pl_v, (
                f"column {name!r} row {i} value divergence: "
                f"pandas {pd_v!r} != polars {pl_v!r}"
            )


def assert_backend_parity(
    make_recipe: Callable[[], Recipe],
    data: Mapping[str, list[Any]],
) -> None:
    """Assert a step produces identical output on pandas and polars.

    Fits and transforms ``make_recipe()`` on the *same* ``data`` materialized as
    both a pandas and a polars frame, then asserts the two narwhals results are
    identical: same column names in the same order, same dtypes, *and* the same
    values (missing-aware, with float columns compared approximately). This
    catches both output-dtype divergence between backends -- the class of bug
    PR #74 fixed for ``StepDummy``, where narwhals' ``lit(None)`` branch unified
    pandas indicator columns to ``Object`` while polars stayed numeric -- and
    value/estimate divergence, where a step computes different numbers depending
    on which backend it fitted and transformed on.

    ``make_recipe`` must return a *fresh, unfitted* recipe on each call so the
    two backends do not share fitted estimates.
    """
    pd_out = make_recipe().fit(pd.DataFrame(data)).transform(pd.DataFrame(data))
    pl_out = make_recipe().fit(pl.DataFrame(data)).transform(pl.DataFrame(data))
    pd_nw = nw.from_native(pd_out, eager_only=True)
    pl_nw = nw.from_native(pl_out, eager_only=True)

    pd_schema, pl_schema = pd_nw.schema, pl_nw.schema
    assert dict(pd_schema) == dict(pl_schema), (
        f"backend dtype divergence: pandas {dict(pd_schema)} != polars {dict(pl_schema)}"
    )

    pd_cols = pd_nw.to_dict(as_series=False)
    pl_cols = pl_nw.to_dict(as_series=False)
    for name in pd_schema:
        _assert_columns_equal(name, pd_cols[name], pl_cols[name])


def fit_transform_isolated(step: Step, data: DataFrame) -> DataFrame:
    """Fit and transform a single step against a stubbed ``SchemaInfo``.

    Exercises the step contract with no ``Recipe`` in the loop: the info is a
    bare ``SchemaInfo.from_frame`` snapshot (no lineage DAG, no role overrides),
    which is all a step reads for its own logic. Returns the native output frame
    so callers can compare its columns against the recipe-driven path.
    """
    frame = _frame.from_native(data)
    info = SchemaInfo.from_frame(frame)
    return _frame.to_native(step.fit(frame, info).transform(frame))
