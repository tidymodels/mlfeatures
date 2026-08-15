"""Conformance tests for StepDummy.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDummy belongs in ``test_step_dummy.py`` and ``test_step_dummy_e2e.py``.
"""

from __future__ import annotations

from shoez import StepDummy
from shoez.testing import StepConformanceTest


class TestDummyConformance(StepConformanceTest):
    training_data = {"g": ["a", "b", "a"], "h": ["x", "y", "x"]}

    def make_step(self) -> StepDummy:
        return StepDummy(["g"])
