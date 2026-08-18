"""Conformance tests for StepIntercept.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepIntercept belongs in ``test_step_intercept.py`` and ``test_step_intercept_e2e.py``.
"""

from __future__ import annotations

from shoez import StepIntercept
from shoez.testing import StepConformanceTest


class TestInterceptConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}

    def make_step(self) -> StepIntercept:
        return StepIntercept()

    # StepIntercept always appends its column and estimates nothing, so it has
    # neither an input selector nor fit-time state. Both checks opt out
    # explicitly rather than being silently absent.
    estimates_attr = None

    def make_empty_step(self) -> None:
        """No column selector, so an empty selection cannot arise."""
        return None
