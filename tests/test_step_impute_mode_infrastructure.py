"""Conformance tests for StepImputeMode.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepImputeMode belongs in ``test_step_impute_mode.py`` and ``test_step_impute_mode_e2e.py``.
"""

from __future__ import annotations

from shoez import StepImputeMode
from shoez.testing import StepConformanceTest


class TestImputeModeConformance(StepConformanceTest):
    training_data = {"g": ["a", None, "b"], "h": ["x", "y", None]}

    def make_step(self) -> StepImputeMode:
        return StepImputeMode(["g"])
