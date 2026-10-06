"""Conformance tests for StepFilter.

The step contract itself lives in `mlfeatures.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepFilter belongs in ``test_step_filter.py`` and ``test_step_filter_e2e.py``.
"""

from __future__ import annotations

from mlfeatures import StepFilter
from mlfeatures.testing import StepConformanceTest


def _keep_over_one(df: object) -> object:
    """Module-level predicate (picklable, unlike a lambda) keeping rows a > 1."""
    return df["a"] > 1  # type: ignore[index]


class TestFilterConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}

    def make_step(self) -> StepFilter:
        return StepFilter(_keep_over_one)

    # StepFilter takes a row predicate, not a column selector, and stores no
    # estimates: it only acts at fit time.
    estimates_attr = None

    def make_empty_step(self) -> None:
        """No column selector, so an empty selection cannot arise."""
        return None
