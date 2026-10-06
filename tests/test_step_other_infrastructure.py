"""Conformance tests for StepOther.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepOther belongs in ``test_step_other.py`` and ``test_step_other_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepOther
from mlfeatures.testing import StepConformanceTest


class TestOtherConformance(StepConformanceTest):
    training_data = {
        "g": ["a", "a", "a", "b", "c"],
        "h": ["x", "y", "x", "y", "z"],
    }

    def make_step(self) -> StepOther:
        return StepOther(["g"], threshold=0.25)
