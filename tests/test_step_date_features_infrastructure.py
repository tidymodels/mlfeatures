"""Conformance tests for StepDateFeatures.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDateFeatures belongs in ``test_step_date_features.py`` and
``test_step_date_features_e2e.py``.

Pandas cannot represent narwhals' ``Date`` dtype from a plain list of
``datetime.date`` objects (it infers ``object`` dtype instead; only a
pyarrow-backed column round-trips as ``Date``). ``dtype_casts`` tells
`StepConformanceTest` to cast the ``"d"`` column to `narwhals.Date` on both
backends after construction, which is what actually produces a ``Date``
column on the pandas side (`StepDateFeatures.required_pkgs` declares the
pyarrow dependency this needs).
"""

from __future__ import annotations

from datetime import date

import narwhals.stable.v2 as nw

from mlfeatures import StepDateFeatures
from mlfeatures.testing import StepConformanceTest


class TestDateFeaturesConformance(StepConformanceTest):
    training_data = {
        "d": [date(2020, 1, 1), date(2020, 6, 15), date(2020, 12, 31)],
        "n": [1.0, 2.0, 3.0],
    }
    dtype_casts = {"d": nw.Date}

    def make_step(self) -> StepDateFeatures:
        return StepDateFeatures(["d"])
