"""Conformance/infrastructure tests for StepImputeMedian.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pandas as pd

from shoez import Recipe, StepImputeMedian


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [2.0, 4.0, None]})
        out = Recipe(steps=[StepImputeMedian([])]).fit(df).transform(df)
        assert list(out.columns) == ["a", "b"]
        pd.testing.assert_frame_equal(out, df)
