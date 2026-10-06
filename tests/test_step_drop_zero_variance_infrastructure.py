"""Conformance tests for StepDropZeroVariance.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDropZeroVariance belongs in ``test_step_drop_zero_variance.py`` and ``test_step_drop_zero_variance_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepDropZeroVariance
from mlfeatures.testing import StepConformanceTest


class TestDropZeroVarianceConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "const": [7.0, 7.0, 7.0]}

    def make_step(self) -> StepDropZeroVariance:
        return StepDropZeroVariance(["a", "const"])
