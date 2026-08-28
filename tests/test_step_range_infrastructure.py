"""Conformance tests for StepRange.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepRange belongs in ``test_step_range.py`` and ``test_step_range_e2e.py``.
"""

from __future__ import annotations

from shoez import StepRange
from shoez.testing import StepConformanceTest


class TestRangeConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepRange:
        return StepRange(["a"])
