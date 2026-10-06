"""Conformance tests for StepUnknown.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepUnknown belongs in ``test_step_unknown.py`` and ``test_step_unknown_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepUnknown
from mlfeatures.testing import StepConformanceTest


class TestUnknownConformance(StepConformanceTest):
    training_data = {"g": ["a", None, "b"], "h": ["x", "y", None]}

    def make_step(self) -> StepUnknown:
        return StepUnknown(["g"])

    # StepUnknown relabels missing values in place; it estimates nothing.
    estimates_attr = None
