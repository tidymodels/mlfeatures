"""Conformance/infrastructure tests for StepIntercept.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.

Note: the empty-selection conformance check does not apply to StepIntercept.
It is an ADDING step that takes no column selector — it always appends a single
constant column regardless of the input columns — so there is no "zero features
selected" state to exercise.
"""

from __future__ import annotations

import pickle

import pandas as pd

from shoez import Recipe, StepIntercept


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    return Recipe(steps=[StepIntercept()]).fit(train), train


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


class TestPickleRoundTrip:
    """A pickled fitted step reproduces the same transform() output."""

    def test_pickle_preserves_transform_output(self) -> None:
        fitted, train = _fitted()
        before = fitted.transform(train)
        restored = pickle.loads(pickle.dumps(fitted))
        pd.testing.assert_frame_equal(restored.transform(train), before)
