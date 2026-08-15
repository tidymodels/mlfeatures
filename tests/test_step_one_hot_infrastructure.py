"""Conformance tests for StepOneHot.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepOneHot belongs in ``test_step_one_hot.py`` and ``test_step_one_hot_e2e.py``.
"""

from __future__ import annotations

from shoez import StepOneHot
from shoez.testing import StepConformanceTest


class TestOneHotConformance(StepConformanceTest):
    training_data = {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}

    def make_step(self) -> StepOneHot:
        return StepOneHot(["g"])
