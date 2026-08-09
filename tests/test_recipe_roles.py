"""Roles + canonical constructor (set_roles, formula sugar).

Covers role assignment on a recipe and the minimal formula parser, both of
which are pure (return a new recipe) and validate against the prototype data.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Recipe, Roles, StepDropCorr, StepNormalize
from shoez import Selectors as s


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "x1": [4.0, 5.0, 6.0],
            "x2": [7.0, 8.0, 9.0],
        }
    )


class TestRoles:
    def test_add_and_query(self) -> None:
        roles = Roles()
        roles.add("a", "predictor")
        roles.add("a", "group")
        assert roles.of("a") == {"predictor", "group"}
        assert roles.columns_with_role("predictor") == ["a"]

    def test_of_missing_is_empty(self) -> None:
        assert Roles().of("nope") == set()

    def test_incompatible_combo_rejected(self, snapshot: SnapshotAssertion) -> None:
        roles = Roles()
        roles.add("a", "outcome")
        with pytest.raises(ValueError) as excinfo:
            roles.add("a", "predictor")
        assert str(excinfo.value) == snapshot

    def test_copy_is_independent(self) -> None:
        roles = Roles({"a": {"predictor"}})
        clone = roles.copy()
        clone.add("a", "group")
        assert roles.of("a") == {"predictor"}

    def test_rejected_add_leaves_no_trace(self) -> None:
        roles = Roles({"a": {"outcome"}, "c": {"predictor"}})
        with pytest.raises(ValueError, match="incompatible roles"):
            roles.add("a", "predictor")
        assert roles.of("a") == {"outcome"}
        assert roles.all_columns() == ["a", "c"]

    def test_equality(self) -> None:
        assert Roles({"a": {"predictor"}}) == Roles({"a": {"predictor"}})
        assert Roles({"a": {"predictor"}}) != Roles({"a": {"outcome"}})

    def test_equality_with_non_roles(self) -> None:
        assert Roles() != "not a Roles"


class TestSetRoles:
    """Assignment of the standard roles."""

    def test_assigns_standard_roles(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        assert rec.roles.of("y") == {"outcome"}
        assert rec.roles.columns_with_role("predictor") == ["x1", "x2"]

    def test_case_weights(self) -> None:
        rec = Recipe(data=_df()).set_roles(case_weights="x2")
        assert rec.roles.of("x2") == {"case_weights"}

    def test_string_and_single_element_list_agree(self) -> None:
        as_string = Recipe(data=_df()).set_roles(outcome="y")
        as_list = Recipe(data=_df()).set_roles(outcome=["y"])
        assert as_string.roles == as_list.roles

    def test_multiple_outcomes(self) -> None:
        # Survival-style: both columns are outcomes (design/roles.md).
        rec = Recipe(data=_df()).set_roles(outcome=["y", "x1"])
        assert rec.roles.columns_with_role("outcome") == ["y", "x1"]

    def test_preserves_insertion_order(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=["x2", "x1"])
        assert rec.roles.columns_with_role("predictor") == ["x2", "x1"]

    def test_no_arguments_is_a_no_op_copy(self) -> None:
        base = Recipe(data=_df()).set_roles(outcome="y")
        same = base.set_roles()
        assert same is not base
        assert same.roles == base.roles

    def test_empty_list_assigns_nothing(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=[])
        assert rec.roles.role_labels() == []

    def test_repeated_column_is_idempotent(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=["x1", "x1"])
        assert rec.roles.columns_with_role("predictor") == ["x1"]

    def test_successive_calls_accumulate(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y").set_roles(predictor=["x1"])
        assert rec.roles.of("y") == {"outcome"}
        assert rec.roles.of("x1") == {"predictor"}

    def test_reassigning_same_role_is_idempotent(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y").set_roles(outcome="y")
        assert rec.roles.of("y") == {"outcome"}

    def test_does_not_remove_existing_roles(self) -> None:
        # set_roles is additive: it never clears a role assigned earlier.
        rec = Recipe(data=_df()).set_roles(predictor="x1").set_roles(predictor="x2")
        assert rec.roles.columns_with_role("predictor") == ["x1", "x2"]

    def test_extends_constructor_assignments(self) -> None:
        rec = Recipe(data=_df(), outcome="y").set_roles(predictor=["x1", "x2"])
        assert rec.roles.of("y") == {"outcome"}
        assert rec.roles.columns_with_role("predictor") == ["x1", "x2"]

    def test_arbitrary_column_names(self) -> None:
        # set_roles passes names through as plain strings: spaces, operator
        # characters, and unicode are all fine (unlike the formula grammar).
        df = pd.DataFrame({"y": [1.0], "my col": [2.0], "a+b": [3.0], "café": [4.0]})
        rec = Recipe(data=df).set_roles(
            outcome="y", predictor=["my col", "a+b", "café"]
        )
        assert set(rec.roles.columns_with_role("predictor")) == {
            "my col",
            "a+b",
            "café",
        }


class TestSetRolesCustomRoles:
    """Arbitrary role names passed as extra keywords."""

    def test_custom_role(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=["x1"], group=["x2"])
        assert rec.roles.of("x2") == {"group"}

    def test_several_custom_roles_at_once(self) -> None:
        rec = Recipe(data=_df()).set_roles(group="x1", id="x2")
        assert rec.roles.of("x1") == {"group"}
        assert rec.roles.of("x2") == {"id"}

    def test_custom_role_stacks_with_standard_role(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor="x1").set_roles(group="x1")
        assert rec.roles.of("x1") == {"predictor", "group"}

    def test_custom_role_stacks_within_one_call(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor="x1", group="x1")
        assert rec.roles.of("x1") == {"predictor", "group"}

    def test_two_custom_roles_on_one_column(self) -> None:
        rec = Recipe(data=_df()).set_roles(group="x1").set_roles(date_split="x1")
        assert rec.roles.of("x1") == {"group", "date_split"}

    def test_case_weights_and_predictor_coexist(self) -> None:
        # design/roles.md calls this combination out explicitly as valid.
        rec = Recipe(data=_df()).set_roles(predictor="x1", case_weights="x1")
        assert rec.roles.of("x1") == {"predictor", "case_weights"}

    def test_custom_role_appears_in_labels(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", group="x1")
        assert rec.roles.role_labels() == ["group", "outcome"]

    def test_custom_role_named_like_a_standard_role_is_the_same_role(self) -> None:
        # ``**custom`` cannot shadow the dedicated keywords, so the only way to
        # reach "outcome" is the keyword itself; a custom "group" stays distinct.
        rec = Recipe(data=_df()).set_roles(outcome="y", group="y")
        assert rec.roles.of("y") == {"outcome", "group"}


class TestSetRolesPurity:
    """set_roles returns a new recipe and never mutates the receiver."""

    def test_is_pure(self) -> None:
        base = Recipe(data=_df())
        extended = base.set_roles(outcome="y")
        assert base.roles.of("y") == set()
        assert extended.roles.of("y") == {"outcome"}
        assert base is not extended

    def test_receiver_roles_object_not_shared(self) -> None:
        base = Recipe(data=_df()).set_roles(outcome="y")
        derived = base.set_roles(predictor="x1")
        assert derived.roles is not base.roles
        assert base.roles.of("x1") == set()

    def test_receiver_unchanged_when_assignment_raises(self) -> None:
        # The clone is built first, so a rejected assignment leaves no partial
        # state on either the receiver or its roles.
        base = Recipe(data=_df()).set_roles(outcome="y")
        with pytest.raises(ValueError):
            base.set_roles(predictor=["x1", "y"])
        assert base.roles.of("x1") == set()
        assert base.roles.of("y") == {"outcome"}

    def test_steps_are_preserved(self) -> None:
        base = Recipe(data=_df()).add(StepDropCorr(["x1", "x2"]))
        derived = base.set_roles(outcome="y")
        assert derived.steps == base.steps

    def test_prototype_is_preserved(self) -> None:
        # The derived recipe still validates against the original prototype.
        rec = Recipe(data=_df()).set_roles(outcome="y")
        with pytest.raises(ValueError):
            rec.set_roles(predictor="missing")

    def test_requirement_overrides_are_preserved(self) -> None:
        rec = (
            Recipe(data=_df())
            .update_role_requirements("id", transform_required=False)
            .set_roles(predictor=["x1"], id="x2")
        )
        fitted = rec.fit(_df())
        # 'x2' carries only the 'id' role, whose requirement was overridden, so
        # new data may omit it.
        result = fitted.transform(_df()[["y", "x1"]])
        assert list(result.columns) == ["y", "x1"]


class TestSetRolesValidation:
    """Rejected columns and rejected role combinations."""

    def test_unknown_column_rejected(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(outcome="missing")
        assert str(excinfo.value) == snapshot

    def test_unknown_column_in_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="'missing' is not in the prototype data"):
            Recipe(data=_df()).set_roles(predictor=["x1", "missing"])

    def test_unknown_column_in_custom_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="'missing' is not in the prototype data"):
            Recipe(data=_df()).set_roles(group="missing")

    def test_incompatible_combo_rejected(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(outcome="y", predictor="y")
        assert str(excinfo.value) == snapshot

    def test_incompatible_combo_across_calls_rejected(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y")
        with pytest.raises(ValueError, match="incompatible roles"):
            rec.set_roles(predictor="y")

    def test_incompatible_combo_against_constructor_rejected(self) -> None:
        rec = Recipe(data=_df(), predictors=["x1"])
        with pytest.raises(ValueError, match="incompatible roles"):
            rec.set_roles(outcome="x1")

    def test_without_prototype_skips_existence_check(self) -> None:
        rec = Recipe().set_roles(outcome="anything")
        assert rec.roles.of("anything") == {"outcome"}

    def test_constructor_spelling_of_predictor_rejected(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # 'predictors' would otherwise land in **custom as a bogus custom role.
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(predictors=["x1"])
        assert str(excinfo.value) == snapshot

    def test_other_plural_spellings_are_ordinary_custom_roles(self) -> None:
        # Only 'predictors' is special-cased; nothing else is second-guessed.
        rec = Recipe(data=_df()).set_roles(outcomes="y")
        assert rec.roles.of("y") == {"outcomes"}

    def test_without_prototype_still_rejects_incompatible_combo(self) -> None:
        with pytest.raises(ValueError, match="incompatible roles"):
            Recipe().set_roles(outcome="a", predictor="a")


class TestSetRolesIntegration:
    """How assigned roles carry through repr, add, fit, and transform."""

    def test_roles_survive_add(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y").add(StepDropCorr(["x1", "x2"]))
        assert rec.roles.of("y") == {"outcome"}
        assert len(rec.steps) == 1

    def test_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        fitted = rec.add(StepDropCorr(["x1", "x2"])).fit(_df())
        assert fitted.roles.of("y") == {"outcome"}

    def test_custom_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=["x1", "x2"], group="y")
        fitted = rec.fit(_df())
        assert fitted.roles.of("y") == {"group"}

    def test_declared_predictors_suppress_auto_assignment(self) -> None:
        # An explicit predictor assignment turns off the fit-time fallback, so
        # the unassigned column stays role-less.
        fitted = Recipe(data=_df()).set_roles(predictor=["x1"]).fit(_df())
        assert fitted.roles.columns_with_role("predictor") == ["x1"]
        assert fitted.roles.of("x2") == set()

    def test_outcome_excluded_from_auto_assigned_predictors(self) -> None:
        fitted = Recipe(data=_df()).set_roles(outcome="y").fit(_df())
        assert fitted.roles.of("y") == {"outcome"}
        assert fitted.roles.columns_with_role("predictor") == ["x1", "x2"]

    def test_case_weights_excluded_from_auto_assigned_predictors(self) -> None:
        fitted = Recipe(data=_df()).set_roles(case_weights="x2").fit(_df())
        assert fitted.roles.columns_with_role("predictor") == ["y", "x1"]

    def test_custom_role_only_column_still_auto_assigned_predictor(self) -> None:
        # Only 'outcome' and 'case_weights' are reserved at fit time; a custom
        # role does not keep a column out of the predictor fallback.
        fitted = Recipe(data=_df()).set_roles(group="x2").fit(_df())
        assert fitted.roles.of("x2") == {"group", "predictor"}

    def test_selectors_see_assigned_roles(self) -> None:
        rec = (
            Recipe(data=_df())
            .set_roles(outcome="y", predictor=["x1", "x2"])
            .add(StepNormalize(s.numeric_predictors()))
        )
        result = rec.fit(_df()).transform(_df())
        # 'y' is an outcome, so normalization left it alone.
        assert result["y"].tolist() == [1.0, 2.0, 3.0]
        assert result["x1"].mean() == pytest.approx(0.0)

    def test_repr_summarises_roles(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        assert repr(rec) == snapshot

    def test_works_on_polars(self) -> None:
        df = pl.DataFrame(_df())
        rec = Recipe(data=df).set_roles(outcome="y", predictor=["x1", "x2"])
        assert rec.roles.of("y") == {"outcome"}
        fitted = rec.fit(df)
        assert fitted.roles.columns_with_role("predictor") == ["x1", "x2"]
