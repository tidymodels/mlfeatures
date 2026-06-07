"""Conformance/infrastructure tests for StepClassDistance.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pandas as pd

from shoez import Recipe, StepClassDistance


class TestEmptySelection:
    """An empty feature selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "out": [0, 0, 1, 1]},
        )
        out = Recipe(steps=[StepClassDistance([], outcome="out")]).fit(df).transform(df)
        assert list(out.columns) == ["a", "out"]
        pd.testing.assert_frame_equal(out, df)
