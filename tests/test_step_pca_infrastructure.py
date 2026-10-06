"""Conformance tests for StepPca.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepPca belongs in ``test_step_pca.py`` and ``test_step_pca_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepPca
from mlfeatures.testing import StepConformanceTest


class TestPcaConformance(StepConformanceTest):
    training_data = {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.1, 3.9, 6.1, 8.0]}

    def make_step(self) -> StepPca:
        return StepPca(["x", "y"], num_comp=2)
