"""Conformance tests for StepDatetimeFeatures.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepDatetimeFeatures belongs in ``test_step_datetime_features.py`` and
``test_step_datetime_features_e2e.py``.
"""

from __future__ import annotations

from datetime import datetime

from mlfeatures import StepDatetimeFeatures
from mlfeatures.testing import StepConformanceTest


class TestDatetimeFeaturesConformance(StepConformanceTest):
    training_data = {
        "dt": [
            datetime(2020, 1, 1, 8, 30, 15),
            datetime(2020, 6, 15, 13, 45, 30),
            datetime(2020, 12, 31, 23, 59, 59),
        ],
        "n": [1.0, 2.0, 3.0],
    }

    def make_step(self) -> StepDatetimeFeatures:
        return StepDatetimeFeatures(["dt"])
