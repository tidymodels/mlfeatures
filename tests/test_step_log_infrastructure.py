"""Step-contract conformance for StepLog.

The checks live in `shoez.testing.StepConformanceTest`; this file supplies only
the step and its data. ``a`` is strictly positive so the out-of-domain warning
does not fire during the contract run; that behaviour is covered in
``test_step_log.py``.
"""

from __future__ import annotations

from shoez import StepLog
from shoez.testing import StepConformanceTest


class TestLogConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepLog:
        return StepLog(["a"])


class TestSignedLogConformance(StepConformanceTest):
    """The contract again under ``signed``, which is a different formula.

    The data spans zero, which the default formula could not take at all.
    """

    training_data = {"a": [-10.0, -0.5, 0.0, 10.0], "g": ["x", "y", "x", "y"]}

    def make_step(self) -> StepLog:
        return StepLog(["a"], signed=True)
