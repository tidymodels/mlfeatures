"""Conformance tests for StepInteract.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepInteract belongs in ``test_step_interact.py`` and ``test_step_interact_e2e.py``.
"""

from __future__ import annotations

from shoez import StepInteract
from shoez.testing import StepConformanceTest


class TestInteractConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]}

    def make_step(self) -> StepInteract:
        return StepInteract(["a", "b"])
