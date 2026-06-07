"""Conformance/infrastructure tests for StepOneHot.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pandas as pd

from shoez import Recipe, StepOneHot


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame({"g": ["a", "b", "a"], "h": ["x", "y", "x"]})
    return Recipe(steps=[StepOneHot(["g"])]).fit(train), train


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"g": ["a", "b", "a"], "h": ["x", "y", "x"]})
        out = Recipe(steps=[StepOneHot([])]).fit(df).transform(df)
        assert list(out.columns) == ["g", "h"]
        pd.testing.assert_frame_equal(out, df)


class TestRowCountEdges:
    """transform() on 0-row and 1-row frames yields schema-correct output."""

    def test_zero_row_transform(self) -> None:
        fitted, train = _fitted()
        expected = list(fitted.transform(train).columns)
        out = fitted.transform(train.head(0))
        assert list(out.columns) == expected
        assert out.shape[0] == 0

    def test_one_row_transform(self) -> None:
        fitted, train = _fitted()
        expected = list(fitted.transform(train).columns)
        out = fitted.transform(train.head(1))
        assert list(out.columns) == expected
        assert out.shape[0] == 1
