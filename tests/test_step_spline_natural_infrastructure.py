"""Conformance tests for StepSplineNatural.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this
file only supplies the step and the data to run it against. Behaviour
specific to StepSplineNatural belongs in ``test_step_spline_natural.py`` and
``test_step_spline_natural_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepSplineNatural
from mlfeatures.testing import StepConformanceTest


class TestSplineNaturalConformance(StepConformanceTest):
    training_data = {"a": list(range(1, 11)), "b": [4.0, 5.0, 6.0, 7.0, 8.0] * 2}

    def make_step(self) -> StepSplineNatural:
        return StepSplineNatural(["a"], deg_free=3)
