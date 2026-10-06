"""Conformance tests for StepImputeMean.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepImputeMean belongs in ``test_step_impute_mean.py`` and ``test_step_impute_mean_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepImputeMean
from mlfeatures.testing import StepConformanceTest


class TestImputeMeanConformance(StepConformanceTest):
    training_data = {"a": [1.0, None, 3.0], "b": [2.0, 4.0, None]}

    def make_step(self) -> StepImputeMean:
        return StepImputeMean(["a", "b"])
