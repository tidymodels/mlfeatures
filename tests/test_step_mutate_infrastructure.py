"""Conformance/infrastructure tests for StepMutate.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.
"""

from __future__ import annotations

import pickle

import pandas as pd

from shoez import Recipe, StepMutate
from tests._conformance import assert_backend_parity, fit_transform_isolated


def _double_a(df: object) -> object:
    """Module-level mutation (picklable, unlike a lambda): a doubled."""
    return df["a"] * 2  # type: ignore[index]


def _mutations() -> dict[str, object]:
    return {"a_doubled": _double_a}


def _fitted() -> tuple[Recipe, pd.DataFrame]:
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    return Recipe(steps=[StepMutate(_mutations())]).fit(train), train


class TestEmptySelection:
    """Not applicable to StepMutate.

    StepMutate is the ``UNCONSTRAINED`` escape hatch and takes no column
    selector — it applies caller-supplied expressions rather than selecting a
    set of columns — so there is no "zero features selected" state to exercise.
    The class is kept as a visible slot (per ``tests/CLAUDE.md``) so the
    omission is documented rather than silently missing.
    """


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
    """Not applicable: StepMutate estimates no statistics — the mutations are
    fixed constructor arguments and ``fit()`` only records which columns they
    add, so there is no ``_estimates`` attribute. Slot kept visible per the
    conformance contract."""


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
            lambda: Recipe(steps=[StepMutate(_mutations())]),
            {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]},
        )


class TestIsolatedFit:
    """A step fits and transforms against a stubbed SchemaInfo, with no Recipe."""

    def test_isolated_fit_matches_recipe(self) -> None:
        fitted, train = _fitted()
        expected = list(fitted.transform(train).columns)
        out = fit_transform_isolated(StepMutate(_mutations()), train)
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
