"""Conformance tests for StepPolynomial.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepPolynomial belongs in ``test_step_polynomial.py`` and ``test_step_polynomial_e2e.py``.
"""

from __future__ import annotations

from shoez import StepPolynomial
from shoez.testing import StepConformanceTest


class TestPolynomialConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}

    def make_step(self) -> StepPolynomial:
        return StepPolynomial(["a"], degree=2)
