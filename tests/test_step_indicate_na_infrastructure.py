"""Conformance tests for StepIndicateNa.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepIndicateNa belongs in ``test_step_indicate_na.py`` and
``test_step_indicate_na_e2e.py``.
"""

from __future__ import annotations

from shoez import StepIndicateNa
from shoez.testing import StepConformanceTest


class TestIndicateNAConformance(StepConformanceTest):
    training_data = {
        "x": [1.0, None, 3.0],
        "g": ["a", "b", None],
    }

    def make_step(self) -> StepIndicateNa:
        return StepIndicateNa(["x"])
