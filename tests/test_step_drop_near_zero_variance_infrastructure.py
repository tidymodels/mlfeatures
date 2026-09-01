"""Conformance tests for StepDropNearZeroVariance.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDropNearZeroVariance belongs in ``test_step_drop_near_zero_variance.py`` and
``test_step_drop_near_zero_variance_e2e.py``.
"""

from __future__ import annotations

from shoez import StepDropNearZeroVariance
from shoez.testing import StepConformanceTest


class TestDropNearZeroVarianceConformance(StepConformanceTest):
    training_data = {
        "a": list(range(21)),
        "sparse": [1.0] + [0.0] * 20,
    }

    def make_step(self) -> StepDropNearZeroVariance:
        return StepDropNearZeroVariance(["a", "sparse"])
