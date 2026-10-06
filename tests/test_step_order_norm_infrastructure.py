"""Conformance tests for StepOrderNorm.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepOrderNorm belongs in ``test_step_order_norm.py`` and
``test_step_order_norm_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepOrderNorm
from mlfeatures.testing import StepConformanceTest


class TestOrderNormConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 4.0, 8.0, 16.0], "g": ["x", "y", "x", "y", "x"]}

    def make_step(self) -> StepOrderNorm:
        return StepOrderNorm(["a"])
