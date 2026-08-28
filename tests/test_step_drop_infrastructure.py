"""Conformance tests for StepDrop.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDrop belongs in ``test_step_drop.py`` and ``test_step_drop_e2e.py``.
"""

from __future__ import annotations

from shoez import StepDrop
from shoez.testing import StepConformanceTest


class TestDropConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "note": ["x", "y", "z"]}

    def make_step(self) -> StepDrop:
        return StepDrop(["note"])
