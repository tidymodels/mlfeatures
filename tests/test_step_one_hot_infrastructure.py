"""Conformance/infrastructure tests for StepOneHot.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pandas as pd

from shoez import Recipe, StepOneHot


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"g": ["a", "b", "a"], "h": ["x", "y", "x"]})
        out = Recipe(steps=[StepOneHot([])]).fit(df).transform(df)
        assert list(out.columns) == ["g", "h"]
        pd.testing.assert_frame_equal(out, df)
