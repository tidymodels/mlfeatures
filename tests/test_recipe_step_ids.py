"""Step-id uniqueness within a recipe.

Step ids key the lineage DAG, ``SchemaInfo.producing_steps``, the ``has_source``
selectors, and step-level error messages, so a recipe must never hold two steps
with the same id. Defaulted ids are disambiguated with a numeric suffix;
explicitly passed duplicates raise.
"""

from __future__ import annotations

import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Recipe, StepDummy, StepNormalize
from shoez import Selectors as s


def _data() -> dict[str, list[float] | list[str]]:
    return {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [4.0, 3.0, 2.0, 1.0],
        "g": ["x", "y", "x", "y"],
        "h": ["p", "q", "p", "q"],
    }


class TestDefaultIdsAreDisambiguated:
    def test_second_same_type_step_is_suffixed(self) -> None:
        recipe = (
            Recipe()
            .add(StepNormalize(s.numeric_features()))
            .add(StepNormalize(s.numeric_features()))
        )
        assert [step.id for step in recipe.steps] == ["normalize", "normalize_2"]

    def test_suffixing_continues_past_two(self) -> None:
        recipe = Recipe().add(
            StepNormalize(["a"]), StepNormalize(["a"]), StepNormalize(["a"])
        )
        assert [step.id for step in recipe.steps] == [
            "normalize",
            "normalize_2",
            "normalize_3",
        ]

    def test_suffixing_skips_an_id_already_taken(self) -> None:
        # The explicit ``normalize_2`` is not rewritten, so the defaulted third
        # step has to step over it.
        recipe = Recipe().add(
            StepNormalize(["a"]),
            StepNormalize(["a"], id="normalize_2"),
            StepNormalize(["a"]),
        )
        assert [step.id for step in recipe.steps] == [
            "normalize",
            "normalize_2",
            "normalize_3",
        ]

    def test_different_types_keep_their_defaults(self) -> None:
        recipe = Recipe().add(StepNormalize(["a"]), StepDummy(["g"]))
        assert [step.id for step in recipe.steps] == ["normalize", "dummy"]

    def test_constructor_steps_are_disambiguated_too(self) -> None:
        recipe = Recipe(steps=[StepNormalize(["a"]), StepNormalize(["a"])])
        assert [step.id for step in recipe.steps] == ["normalize", "normalize_2"]

    def test_renaming_does_not_mutate_the_callers_step(self) -> None:
        step = StepNormalize(["a"])
        recipe = Recipe().add(StepNormalize(["a"])).add(step)
        assert step.id == "normalize"
        assert recipe.steps[1].id == "normalize_2"
        assert recipe.steps[1] is not step

    def test_clone_is_idempotent(self) -> None:
        # ``add`` clones the recipe, which re-runs id resolution over the
        # already-resolved steps; a second pass must not re-suffix them.
        recipe = (
            Recipe()
            .add(StepNormalize(["a"]))
            .add(StepNormalize(["a"]))
            .add(StepDummy(["g"]))
        )
        assert [step.id for step in recipe.steps] == [
            "normalize",
            "normalize_2",
            "dummy",
        ]


class TestExplicitDuplicatesRaise:
    def test_explicit_duplicate_of_an_explicit_id(
        self, snapshot: SnapshotAssertion
    ) -> None:
        recipe = Recipe().add(StepNormalize(["a"], id="scale"))
        with pytest.raises(ValueError) as excinfo:
            recipe.add(StepDummy(["g"], id="scale"))
        assert str(excinfo.value) == snapshot

    def test_explicit_duplicate_of_a_defaulted_id(
        self, snapshot: SnapshotAssertion
    ) -> None:
        recipe = Recipe().add(StepNormalize(["a"]))
        with pytest.raises(ValueError) as excinfo:
            recipe.add(StepNormalize(["b"], id="normalize"))
        assert str(excinfo.value) == snapshot

    def test_explicit_duplicate_within_one_add_call(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe().add(StepNormalize(["a"], id="scale"), StepDummy(["g"], id="scale"))
        assert str(excinfo.value) == snapshot

    def test_explicit_duplicate_in_the_constructor(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(
                steps=[StepNormalize(["a"], id="scale"), StepDummy(["g"], id="scale")]
            )
        assert str(excinfo.value) == snapshot


class TestProvenanceIsUnambiguous:
    def test_columns_from_source_separates_two_dummy_steps(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(data=df, features=["a", "b", "g", "h"]).add(
            StepDummy(["g"]), StepDummy(["h"])
        )
        fitted = recipe.fit(df)

        first = fitted.schema.columns_from_source("dummy")
        second = fitted.schema.columns_from_source("dummy_2")
        assert first and second
        assert not set(first) & set(second)
        assert all(name.startswith("g_") for name in first)
        assert all(name.startswith("h_") for name in second)

    def test_has_source_selects_only_the_named_step(self) -> None:
        df = pd.DataFrame(_data())
        recipe = (
            Recipe(data=df, features=["a", "b", "g", "h"])
            .add(StepDummy(["g"]), StepDummy(["h"]))
            .add(StepNormalize(s.has_source("dummy")))
        )
        fitted = recipe.fit(df)
        normalize = fitted.steps[-1]
        selected = normalize.schema_in().columns
        assert selected
        assert all(name.startswith("g_") for name in selected)

    def test_producing_steps_keeps_both_steps(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(data=df, features=["a", "b", "g", "h"]).add(
            StepDummy(["g"]), StepDummy(["h"]), StepNormalize(["a"])
        )
        fitted = recipe.fit(df)
        assert [step.id for step in fitted.steps] == [
            "dummy",
            "dummy_2",
            "normalize",
        ]
