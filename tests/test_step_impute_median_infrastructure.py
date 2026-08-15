"""Conformance tests for StepImputeMedian.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepImputeMedian belongs in ``test_step_impute_median.py`` and ``test_step_impute_median_e2e.py``.
"""

from __future__ import annotations

from shoez import StepImputeMedian
from shoez.testing import StepConformanceTest


class TestImputeMedianConformance(StepConformanceTest):
    training_data = {"a": [1.0, None, 3.0], "b": [2.0, 4.0, None]}

    def make_step(self) -> StepImputeMedian:
        return StepImputeMedian(["a", "b"])
