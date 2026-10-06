"""Conformance tests for StepClassDistance.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepClassDistance belongs in ``test_step_class_distance.py`` and ``test_step_class_distance_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepClassDistance
from mlfeatures.testing import StepConformanceTest


class TestClassDistanceConformance(StepConformanceTest):
    training_data = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
        "out": [0, 0, 1, 1],
    }

    def make_step(self) -> StepClassDistance:
        return StepClassDistance(["a", "b"], target="out")
