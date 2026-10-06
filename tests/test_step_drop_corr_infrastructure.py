"""Conformance tests for StepDropCorr.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDropCorr belongs in ``test_step_drop_corr.py`` and ``test_step_drop_corr_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepDropCorr
from mlfeatures.testing import StepConformanceTest


class TestDropCorrConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]}

    def make_step(self) -> StepDropCorr:
        return StepDropCorr(["a", "b"])
