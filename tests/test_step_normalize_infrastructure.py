"""Conformance/infrastructure tests for StepNormalize.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pandas as pd

from shoez import Recipe, StepNormalize


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
        out = Recipe(steps=[StepNormalize([])]).fit(df).transform(df)
        assert list(out.columns) == ["a", "g"]
        pd.testing.assert_frame_equal(out, df)
