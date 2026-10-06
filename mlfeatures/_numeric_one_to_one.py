"""Shared ``schema_out()`` logic for the ONE_TO_ONE numeric-transform steps.

`StepNormalize`, `StepRange`, `StepLog`, and `StepOrderNorm` each replace a
subset of their selected columns in place with a transformed value and leave
the rest untouched (a constant column `StepNormalize`/`StepRange` can't
rescale, a non-positive value `StepLog` can't take the logarithm of, ...).
Whichever columns a step actually transforms are promoted to ``Float64``;
every other selected column keeps its input dtype. This module holds that
one shared computation; each step still owns its own `_estimates` (whatever
per-column state it needs) and only has to say *which* columns it touched.

Not part of the public API: only ``mlfeatures.step_normalize``, ``mlfeatures.step_range``,
``mlfeatures.step_log``, and ``mlfeatures.step_order_norm`` import from here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlfeatures.types import ColSchema

if TYPE_CHECKING:
    from collections.abc import Collection


def schema_out_for_touched(schema_in: ColSchema, touched: Collection[str]) -> ColSchema:
    """``schema_out()`` after a ONE_TO_ONE numeric transform: touched columns become Float64.

    Every column in ``touched`` becomes ``Float64``; every other column of
    ``schema_in`` is left unchanged and keeps its input dtype.
    """
    dtypes = {
        col: ("Float64" if col in touched else schema_in.dtypes[col])
        for col in schema_in.columns
    }
    return ColSchema(columns=schema_in.columns, dtypes=dtypes)
