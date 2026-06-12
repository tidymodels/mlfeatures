"""Conformance/infrastructure tests for StepUnknown.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pickle

import pandas as pd

from shoez import Recipe, StepUnknown


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame({"g": ["a", None, "b"], "h": ["x", "y", None]})
    return Recipe(steps=[StepUnknown(["g"])]).fit(train), train


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"g": ["a", None, "b"], "h": ["x", "y", None]})
        out = Recipe(steps=[StepUnknown([])]).fit(df).transform(df)
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


class TestEstimatesLifecycle:
    """Not applicable: StepUnknown estimates no statistics — its recoding is
    parameter-driven and it stores only its input schema, so there is no
    ``_estimates`` attribute. Slot kept visible per the conformance contract."""


class TestLineage:
    """output_parents() declares exactly the columns the step adds, each with valid parents."""

    def test_output_parents_matches_added_columns(self) -> None:
        fitted, train = _fitted()
        step = fitted.steps[0]
        pre = set(train.columns)
        added = set(fitted.transform(train).columns) - pre
        parents = step.output_parents()
        assert set(parents) == added
        assert all(
            parent in pre for parents_of in parents.values() for parent in parents_of
        )
