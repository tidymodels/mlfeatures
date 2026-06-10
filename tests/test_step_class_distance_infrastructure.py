"""Conformance/infrastructure tests for StepClassDistance.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pickle

import pandas as pd

from shoez import Recipe, StepClassDistance


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "out": [0, 0, 1, 1]}
    )
    return Recipe(steps=[StepClassDistance(["a", "b"], outcome="out")]).fit(
        train
    ), train


class TestEmptySelection:
    """An empty feature selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "out": [0, 0, 1, 1]},
        )
        out = Recipe(steps=[StepClassDistance([], outcome="out")]).fit(df).transform(df)
        assert list(out.columns) == ["a", "out"]
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


class TestPickleRoundTrip:
    """A pickled fitted step reproduces the same transform() output."""

    def test_pickle_preserves_transform_output(self) -> None:
        fitted, train = _fitted()
        before = fitted.transform(train)
        restored = pickle.loads(pickle.dumps(fitted))
        pd.testing.assert_frame_equal(restored.transform(train), before)


class TestFitDeterminism:
    """Fitting two independent recipes on the same data yields identical transform output."""

    def test_fit_is_deterministic(self) -> None:
        first, train = _fitted()
        second, _ = _fitted()
        pd.testing.assert_frame_equal(first.transform(train), second.transform(train))


class TestStableId:
    """The same step type with the same hyperparameters yields the same id."""

    def test_id_is_stable(self) -> None:
        first, _ = _fitted()
        second, _ = _fitted()
        assert first.steps[0].id == second.steps[0].id
