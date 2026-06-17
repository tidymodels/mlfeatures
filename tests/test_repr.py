"""Printing methods for recipes, fitted recipes, steps, and selectors.

Covers the ``repr`` layout described in ``design/inspection.md``: the
unprepped/prepped distinction, the role-summary line, aligned step markers, and
the resolved-vs-unresolved selector display.

The exact rendered output is pinned with syrupy snapshots (run
``pytest --snapshot-update`` to refresh after an intentional layout change).
"""

from __future__ import annotations

import pandas as pd
from syrupy.assertion import SnapshotAssertion

from shoez import Recipe, StepDummy, StepIntercept, StepNormalize
from shoez import Selectors as cs


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 22.5, 7.0, 30.0],
            "age": [25, 41, 33, 52],
            "city": ["NYC", "LA", "NYC", "SF"],
        }
    )


class TestSelectorRepr:
    """Selectors print as their recognisable factory call, never as ``<object>``."""

    def test_name_selectors(self, snapshot: SnapshotAssertion) -> None:
        assert {
            "starts_with": repr(cs.starts_with("x")),
            "ends_with": repr(cs.ends_with("_id")),
            "contains": repr(cs.contains("z")),
            "matches": repr(cs.matches("^x")),
            "col": repr(cs.col("a", "b")),
        } == snapshot

    def test_role_and_type_selectors(self, snapshot: SnapshotAssertion) -> None:
        assert {
            "everything": repr(cs.everything()),
            "predictors": repr(cs.predictors()),
            "numeric_predictors": repr(cs.numeric_predictors()),
            "nominal": repr(cs.nominal()),
        } == snapshot

    def test_composite_and_complement(self, snapshot: SnapshotAssertion) -> None:
        assert {
            "and": repr(cs.predictors() & cs.numeric()),
            "or": repr(cs.predictors() | cs.numeric()),
            "complement": repr(~cs.predictors()),
        } == snapshot

    def test_source_selectors(self, snapshot: SnapshotAssertion) -> None:
        assert {
            "has_source": repr(cs.has_source("dummy")),
            "has_ancestor": repr(cs.has_ancestor(step_type=StepDummy)),
        } == snapshot


class TestStepRepr:
    """A step prints its snake-case type, selection, and training state."""

    def test_untrained_shows_unresolved_selector(
        self, snapshot: SnapshotAssertion
    ) -> None:
        assert repr(StepNormalize(cs.numeric_predictors())) == snapshot

    def test_untrained_list_selector(self, snapshot: SnapshotAssertion) -> None:
        assert repr(StepDummy(["city"])) == snapshot

    def test_trained_shows_resolved_columns(self, snapshot: SnapshotAssertion) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(cs.numeric_predictors()))
            .fit(_df())
        )
        (step,) = rec.steps
        assert repr(step) == snapshot

    def test_intercept_consumes_no_columns(self, snapshot: SnapshotAssertion) -> None:
        assert repr(StepIntercept()) == snapshot

    def test_trained_empty_selection(self, snapshot: SnapshotAssertion) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(cs.starts_with("zzz")))
            .fit(_df())
        )
        (step,) = rec.steps
        assert repr(step) == snapshot


class TestRecipeRepr:
    """The unprepped recipe view."""

    def test_empty_recipe(self, snapshot: SnapshotAssertion) -> None:
        assert repr(Recipe()) == snapshot

    def test_roles_and_column_count(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df(), outcome="price", predictors=["age", "city"])
        assert repr(rec) == snapshot

    def test_steps_listed_untrained(self, snapshot: SnapshotAssertion) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
        )
        assert repr(rec) == snapshot

    def test_no_prototype_omits_column_count(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe().add(StepNormalize(["age"]))
        assert repr(rec) == snapshot

    def test_markers_are_aligned(self) -> None:
        # Structural property (not a content snapshot): every training-state
        # marker must start at the same column so the listing lines up.
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
        )
        marker_cols = [
            line.index("[untrained]")
            for line in repr(rec).splitlines()
            if "[untrained]" in line
        ]
        assert len(marker_cols) == 2
        assert len(set(marker_cols)) == 1


class TestFittedRecipeRepr:
    """The prepped recipe view."""

    def test_prepped_header_and_trained_markers(
        self, snapshot: SnapshotAssertion
    ) -> None:
        fitted = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
            .fit(_df())
        )
        assert repr(fitted) == snapshot

    def test_resolved_columns_shown(self, snapshot: SnapshotAssertion) -> None:
        fitted = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .fit(_df())
        )
        assert repr(fitted) == snapshot
