"""Printing methods for recipes, fitted recipes, steps, and selectors.

Covers the ``repr`` layout described in ``design/inspection.md``: the
unprepped/prepped distinction, the role-summary line, aligned step markers, and
the resolved-vs-unresolved selector display.
"""

from __future__ import annotations

import pandas as pd

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

    def test_name_selectors(self) -> None:
        assert repr(cs.starts_with("x")) == "starts_with('x')"
        assert repr(cs.ends_with("_id")) == "ends_with('_id')"
        assert repr(cs.contains("z")) == "contains('z')"
        assert repr(cs.matches("^x")) == "matches('^x')"
        assert repr(cs.col("a", "b")) == "col('a', 'b')"

    def test_role_and_type_selectors(self) -> None:
        assert repr(cs.everything()) == "everything()"
        assert repr(cs.predictors()) == "select(role='predictor')"
        assert "role='predictor'" in repr(cs.numeric_predictors())
        assert "type=" in repr(cs.numeric_predictors())
        assert repr(cs.nominal()).startswith("select(type=(")

    def test_composite_and_complement(self) -> None:
        assert repr(cs.predictors() & cs.numeric()) == (
            "select(role='predictor') & select(type=NumericType)"
        )
        assert repr(cs.predictors() | cs.numeric()).count(" | ") == 1
        assert repr(~cs.predictors()) == "~select(role='predictor')"

    def test_source_selectors(self) -> None:
        assert repr(cs.has_source("dummy")) == "has_source('dummy')"
        assert repr(cs.has_ancestor(step_type=StepDummy)) == (
            "has_ancestor(step_type=StepDummy)"
        )


class TestStepRepr:
    """A step prints its snake-case type, selection, and training state."""

    def test_untrained_shows_unresolved_selector(self) -> None:
        step = StepNormalize(cs.numeric_predictors())
        text = repr(step)
        assert text.startswith("step_normalize(")
        assert text.endswith("[untrained]")
        assert "role='predictor'" in text

    def test_untrained_list_selector(self) -> None:
        assert repr(StepDummy(["city"])) == "step_dummy(['city'])  [untrained]"

    def test_type_label_is_snake_case(self) -> None:
        assert repr(StepDummy(["city"])).startswith("step_dummy(")

    def test_trained_shows_resolved_columns(self) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(cs.numeric_predictors()))
            .fit(_df())
        )
        (step,) = rec.steps
        assert repr(step) == "step_normalize(age)  [trained]"

    def test_intercept_consumes_no_columns(self) -> None:
        assert repr(StepIntercept()) == "step_intercept()  [untrained]"

    def test_trained_empty_selection(self) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(cs.starts_with("zzz")))
            .fit(_df())
        )
        (step,) = rec.steps
        assert repr(step) == "step_normalize(<empty selection>)  [trained]"


class TestRecipeRepr:
    """The unprepped recipe view."""

    def test_empty_recipe(self) -> None:
        text = repr(Recipe())
        assert "Recipe (unprepped)" in text
        assert "no roles assigned" in text
        assert "<none>" in text

    def test_roles_and_column_count(self) -> None:
        rec = Recipe(data=_df(), outcome="price", predictors=["age", "city"])
        text = repr(rec)
        assert "Recipe (unprepped)" in text
        assert "3 columns" in text
        assert "1 outcome" in text
        assert "2 predictors" in text

    def test_steps_listed_untrained(self) -> None:
        rec = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
        )
        text = repr(rec)
        assert text.count("[untrained]") == 2
        assert "step_normalize" in text
        assert "step_dummy" in text

    def test_no_prototype_omits_column_count(self) -> None:
        rec = Recipe().add(StepNormalize(["age"]))
        text = repr(rec)
        assert "columns" not in text
        assert "no roles assigned" in text
        assert "step_normalize" in text

    def test_markers_are_aligned(self) -> None:
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

    def test_prepped_header_and_trained_markers(self) -> None:
        fitted = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .add(StepDummy(["city"]))
            .fit(_df())
        )
        text = repr(fitted)
        assert "Recipe (prepped)" in text
        assert text.count("[trained]") == 2
        assert "[untrained]" not in text

    def test_resolved_columns_shown(self) -> None:
        fitted = (
            Recipe(data=_df(), outcome="price", predictors=["age", "city"])
            .add(StepNormalize(["age"]))
            .fit(_df())
        )
        assert "step_normalize(age)" in repr(fitted)
