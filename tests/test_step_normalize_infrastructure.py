"""Conformance tests for StepNormalize.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepNormalize belongs in ``test_step_normalize.py`` and ``test_step_normalize_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepNormalize
from mlfeatures.testing import StepConformanceTest


class TestNormalizeConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepNormalize:
        return StepNormalize(["a"])
