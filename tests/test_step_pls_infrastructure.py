"""Conformance tests for StepPls.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepPls belongs in ``test_step_pls.py`` and ``test_step_pls_e2e.py``.
"""

from __future__ import annotations

from shoez import StepPls
from shoez.testing import StepConformanceTest


class TestPlsConformance(StepConformanceTest):
    training_data = {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [2.0, 4.0, 6.0, 8.0],
        "out": [1.0, 2.0, 2.5, 4.5],
    }

    def make_step(self) -> StepPls:
        return StepPls(["a", "b"], target="out", num_comp=1)
