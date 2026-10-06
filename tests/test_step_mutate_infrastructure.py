"""Conformance tests for StepMutate.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepMutate belongs in ``test_step_mutate.py`` and ``test_step_mutate_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepMutate
from mlfeatures.testing import StepConformanceTest


def _double_a(df: object) -> object:
    """Module-level mutation (picklable, unlike a lambda)."""
    return df["a"] * 2  # type: ignore[index,operator]


class TestMutateConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}

    def make_step(self) -> StepMutate:
        return StepMutate({"a_doubled": _double_a})

    # StepMutate is the UNCONSTRAINED escape hatch: no column selector, no
    # estimates, and its user callables are needed at transform time, so JSON
    # serialization is refused rather than silently dropping them.
    estimates_attr = None
    supports_json = False

    def make_empty_step(self) -> None:
        """No column selector, so an empty selection cannot arise."""
        return None
