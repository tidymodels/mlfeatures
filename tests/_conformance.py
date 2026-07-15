"""Shared helpers for the per-step conformance/infrastructure suite.

Kept in an underscore-prefixed module so pytest does not collect it as a test
file; the ``test_step_x_infrastructure.py`` files import from here to avoid
duplicating cross-cutting conformance logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping

    from shoez import Recipe


def assert_backend_dtype_parity(
    make_recipe: Callable[[], Recipe],
    data: Mapping[str, list[Any]],
) -> None:
    """Assert a step produces identical output schemas on pandas and polars.

    Fits and transforms ``make_recipe()`` on the *same* ``data`` materialized as
    both a pandas and a polars frame, then asserts the two narwhals schemas are
    identical (same column names in the same order, same dtypes). This catches
    output-dtype divergence between backends — the class of bug PR #74 fixed for
    ``StepDummy``, where narwhals' ``lit(None)`` branch unified pandas indicator
    columns to ``Object`` while polars stayed numeric.

    ``make_recipe`` must return a *fresh, unfitted* recipe on each call so the
    two backends do not share fitted estimates.
    """
    pd_out = make_recipe().fit(pd.DataFrame(data)).transform(pd.DataFrame(data))
    pl_out = make_recipe().fit(pl.DataFrame(data)).transform(pl.DataFrame(data))
    pd_schema = nw.from_native(pd_out, eager_only=True).schema
    pl_schema = nw.from_native(pl_out, eager_only=True).schema
    assert dict(pd_schema) == dict(pl_schema), (
        f"backend dtype divergence: pandas {dict(pd_schema)} != polars {dict(pl_schema)}"
    )
