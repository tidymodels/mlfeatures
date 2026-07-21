"""Conformance/infrastructure tests for StepNormalize.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pickle

import pandas as pd

from shoez import Recipe, StepNormalize
from tests._conformance import assert_backend_parity, fit_transform_isolated


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
    return Recipe(steps=[StepNormalize(["a"])]).fit(train), train


class TestEmptySelection:
    """An empty column selection is a no-op: the frame passes through unchanged."""

    def test_empty_selection_is_noop(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
        out = Recipe(steps=[StepNormalize([])]).fit(df).transform(df)
        assert list(out.columns) == ["a", "g"]
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
    """`_estimates` is None before fit() and populated after (generic fitted-state contract)."""

    def test_estimates_populated_after_fit(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]})
        step = StepNormalize(["a"])
        assert step._estimates is None
        Recipe(steps=[step]).fit(train)
        assert step._estimates is not None


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


class TestBackendParity:
    """fit + transform yields identical narwhals output (schema and values) on pandas and polars."""

    def test_pandas_and_polars_output_match(self) -> None:
        assert_backend_parity(
            lambda: Recipe(steps=[StepNormalize(["a"])]),
            {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]},
        )


class TestIsolatedFit:
    """A step fits and transforms against a stubbed SchemaInfo, with no Recipe."""

    def test_isolated_fit_matches_recipe(self) -> None:
        fitted, train = _fitted()
        expected = list(fitted.transform(train).columns)
        out = fit_transform_isolated(StepNormalize(["a"]), train)
        assert list(out.columns) == expected


class TestTransformIdempotent:
    """transform() is pure: re-transforming the same frame yields identical output."""

    def test_transform_is_repeatable(self) -> None:
        fitted, train = _fitted()
        pd.testing.assert_frame_equal(fitted.transform(train), fitted.transform(train))


class TestSparseContract:
    """produces_sparse is a bool; no current step declares sparse output, so the
    sparse-in/sparse-out path is a no-op until one does."""

    def test_produces_sparse_is_bool(self) -> None:
        fitted, _ = _fitted()
        assert isinstance(fitted.steps[0].produces_sparse, bool)


class TestJsonRoundTrip:
    """A fitted step serializes to diffable JSON and reloads to identical
    transform() output.

    This doubles as the no-closure guarantee (``design/serialization.md``): a
    closure cannot be JSON-encoded, so a passing round-trip proves the step and
    its selectors captured only plain data, no environment or globals."""

    def test_json_preserves_transform_output(self) -> None:
        fitted, train = _fitted()
        before = fitted.transform(train)
        restored = fitted.from_json(fitted.to_json())
        pd.testing.assert_frame_equal(restored.transform(train), before)
