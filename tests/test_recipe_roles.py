"""Roles + canonical constructor (set_roles, formula sugar).

Covers role assignment on a recipe and the minimal formula parser, both of
which are pure (return a new recipe) and validate against the prototype data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import narwhals.stable.v2 as nw
import narwhals.stable.v2.selectors as ncs
import pandas as pd
import polars as pl
import polars.selectors as cs
import pytest
from sklearn.base import clone
from syrupy.assertion import SnapshotAssertion

from mlfeatures import (
    FittedRecipe,
    Recipe,
    RecipeTransformer,
    Roles,
    StepDropCorr,
    StepDummy,
    StepFilter,
    StepIntercept,
    StepNormalize,
    _frame,
)
from mlfeatures import Selectors as s


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
        roles.add("a", "feature")
        roles.add("a", "group")
        assert roles.of("a") == {"feature", "group"}
        assert roles.columns_with_role("feature") == ["a"]

    def test_of_missing_is_empty(self) -> None:
        assert Roles().of("nope") == set()

    def test_incompatible_combo_rejected(self, snapshot: SnapshotAssertion) -> None:
        roles = Roles()
        roles.add("a", "target")
        with pytest.raises(ValueError) as excinfo:
            roles.add("a", "feature")
        assert str(excinfo.value) == snapshot

    def test_copy_is_independent(self) -> None:
        roles = Roles({"a": {"feature"}})
        clone = roles.copy()
        clone.add("a", "group")
        assert roles.of("a") == {"feature"}

    def test_rejected_add_leaves_no_trace(self) -> None:
        roles = Roles({"a": {"target"}, "c": {"feature"}})
        with pytest.raises(ValueError, match="incompatible roles"):
            roles.add("a", "feature")
        assert roles.of("a") == {"target"}
        assert roles.all_columns() == ["a", "c"]

    def test_equality(self) -> None:
        assert Roles({"a": {"feature"}}) == Roles({"a": {"feature"}})
        assert Roles({"a": {"feature"}}) != Roles({"a": {"target"}})

    def test_equality_with_non_roles(self) -> None:
        assert Roles() != "not a Roles"


class TestSetRoles:
    """Assignment of the standard roles."""

    def test_assigns_standard_roles(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y", feature=["x1", "x2"])
        assert rec.roles.of("y") == {"target"}
        assert rec.roles.columns_with_role("feature") == ["x1", "x2"]

    def test_case_weights(self) -> None:
        rec = Recipe(data=_df()).set_roles(case_weights="x2")
        assert rec.roles.of("x2") == {"case_weights"}

    def test_string_and_single_element_list_agree(self) -> None:
        as_string = Recipe(data=_df()).set_roles(target="y")
        as_list = Recipe(data=_df()).set_roles(target=["y"])
        assert as_string.roles == as_list.roles

    def test_multiple_targets(self) -> None:
        # Survival-style: both columns are targets (design/roles.md).
        rec = Recipe(data=_df()).set_roles(target=["y", "x1"])
        assert rec.roles.columns_with_role("target") == ["y", "x1"]

    def test_preserves_insertion_order(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature=["x2", "x1"])
        assert rec.roles.columns_with_role("feature") == ["x2", "x1"]

    def test_no_arguments_is_a_no_op_copy(self) -> None:
        base = Recipe(data=_df()).set_roles(target="y")
        same = base.set_roles()
        assert same is not base
        assert same.roles == base.roles

    def test_empty_list_assigns_nothing(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature=[])
        assert rec.roles.role_names() == []

    def test_repeated_column_is_idempotent(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature=["x1", "x1"])
        assert rec.roles.columns_with_role("feature") == ["x1"]

    def test_successive_calls_accumulate(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y").set_roles(feature=["x1"])
        assert rec.roles.of("y") == {"target"}
        assert rec.roles.of("x1") == {"feature"}

    def test_reassigning_same_role_is_idempotent(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y").set_roles(target="y")
        assert rec.roles.of("y") == {"target"}

    def test_does_not_remove_existing_roles(self) -> None:
        # set_roles is additive: it never clears a role assigned earlier.
        rec = Recipe(data=_df()).set_roles(feature="x1").set_roles(feature="x2")
        assert rec.roles.columns_with_role("feature") == ["x1", "x2"]

    def test_extends_constructor_assignments(self) -> None:
        rec = Recipe(data=_df(), target="y").set_roles(feature=["x1", "x2"])
        assert rec.roles.of("y") == {"target"}
        assert rec.roles.columns_with_role("feature") == ["x1", "x2"]

    def test_arbitrary_column_names(self) -> None:
        # set_roles passes names through as plain strings: spaces, operator
        # characters, and unicode are all fine (unlike the formula grammar).
        df = pd.DataFrame({"y": [1.0], "my col": [2.0], "a+b": [3.0], "café": [4.0]})
        rec = Recipe(data=df).set_roles(target="y", feature=["my col", "a+b", "café"])
        assert set(rec.roles.columns_with_role("feature")) == {
            "my col",
            "a+b",
            "café",
        }


class TestSetRolesCustomRoles:
    """Arbitrary role names passed as extra keywords."""

    def test_custom_role(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature=["x1"], group=["x2"])
        assert rec.roles.of("x2") == {"group"}

    def test_several_custom_roles_at_once(self) -> None:
        rec = Recipe(data=_df()).set_roles(group="x1", id="x2")
        assert rec.roles.of("x1") == {"group"}
        assert rec.roles.of("x2") == {"id"}

    def test_custom_role_stacks_with_standard_role(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature="x1").set_roles(group="x1")
        assert rec.roles.of("x1") == {"feature", "group"}

    def test_custom_role_stacks_within_one_call(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature="x1", group="x1")
        assert rec.roles.of("x1") == {"feature", "group"}

    def test_two_custom_roles_on_one_column(self) -> None:
        rec = Recipe(data=_df()).set_roles(group="x1").set_roles(date_split="x1")
        assert rec.roles.of("x1") == {"group", "date_split"}

    def test_case_weights_and_feature_coexist(self) -> None:
        # design/roles.md calls this combination out explicitly as valid.
        rec = Recipe(data=_df()).set_roles(feature="x1", case_weights="x1")
        assert rec.roles.of("x1") == {"feature", "case_weights"}

    def test_custom_role_appears_in_labels(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y", group="x1")
        assert rec.roles.role_names() == ["group", "target"]

    def test_custom_role_named_like_a_standard_role_is_the_same_role(self) -> None:
        # ``**custom`` cannot shadow the dedicated keywords, so the only way to
        # reach "target" is the keyword itself; a custom "group" stays distinct.
        rec = Recipe(data=_df()).set_roles(target="y", group="y")
        assert rec.roles.of("y") == {"target", "group"}


class TestSetRolesPurity:
    """set_roles returns a new recipe and never mutates the receiver."""

    def test_is_pure(self) -> None:
        base = Recipe(data=_df())
        extended = base.set_roles(target="y")
        assert base.roles.of("y") == set()
        assert extended.roles.of("y") == {"target"}
        assert base is not extended

    def test_receiver_roles_object_not_shared(self) -> None:
        base = Recipe(data=_df()).set_roles(target="y")
        derived = base.set_roles(feature="x1")
        assert derived.roles is not base.roles
        assert base.roles.of("x1") == set()

    def test_receiver_unchanged_when_assignment_raises(self) -> None:
        # The clone is built first, so a rejected assignment leaves no partial
        # state on either the receiver or its roles.
        base = Recipe(data=_df()).set_roles(target="y")
        with pytest.raises(ValueError):
            base.set_roles(feature=["x1", "y"])
        assert base.roles.of("x1") == set()
        assert base.roles.of("y") == {"target"}

    def test_steps_are_preserved(self) -> None:
        base = Recipe(data=_df()).add(StepDropCorr(["x1", "x2"]))
        derived = base.set_roles(target="y")
        assert derived.steps == base.steps

    def test_prototype_is_preserved(self) -> None:
        # The derived recipe still validates against the original prototype.
        rec = Recipe(data=_df()).set_roles(target="y")
        with pytest.raises(ValueError):
            rec.set_roles(feature="missing")

    def test_requirement_overrides_are_preserved(self) -> None:
        rec = (
            Recipe(data=_df())
            .update_role_requirements("id", transform_required=False)
            .set_roles(feature=["x1"], id="x2")
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
            Recipe(data=_df()).set_roles(target="missing")
        assert str(excinfo.value) == snapshot

    def test_unknown_column_in_list_rejected(self) -> None:
        with pytest.raises(ValueError, match="'missing' is not in the prototype data"):
            Recipe(data=_df()).set_roles(feature=["x1", "missing"])

    def test_unknown_column_in_custom_role_rejected(self) -> None:
        with pytest.raises(ValueError, match="'missing' is not in the prototype data"):
            Recipe(data=_df()).set_roles(group="missing")

    def test_incompatible_combo_rejected(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(target="y", feature="y")
        assert str(excinfo.value) == snapshot

    def test_incompatible_combo_across_calls_rejected(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y")
        with pytest.raises(ValueError, match="incompatible roles"):
            rec.set_roles(feature="y")

    def test_incompatible_combo_against_constructor_rejected(self) -> None:
        rec = Recipe(data=_df(), features=["x1"])
        with pytest.raises(ValueError, match="incompatible roles"):
            rec.set_roles(target="x1")

    def test_without_prototype_skips_existence_check(self) -> None:
        rec = Recipe().set_roles(target="anything")
        assert rec.roles.of("anything") == {"target"}

    def test_constructor_spelling_of_feature_rejected(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # 'features' would otherwise land in **custom as a bogus custom role.
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(features=["x1"])
        assert str(excinfo.value) == snapshot

    def test_other_plural_spellings_are_ordinary_custom_roles(self) -> None:
        # Only 'features' is special-cased; nothing else is second-guessed.
        rec = Recipe(data=_df()).set_roles(targets="y")
        assert rec.roles.of("y") == {"targets"}

    def test_without_prototype_still_rejects_incompatible_combo(self) -> None:
        with pytest.raises(ValueError, match="incompatible roles"):
            Recipe().set_roles(target="a", feature="a")


class TestSetRolesIntegration:
    """How assigned roles carry through repr, add, fit, and transform."""

    def test_roles_survive_add(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y").add(StepDropCorr(["x1", "x2"]))
        assert rec.roles.of("y") == {"target"}
        assert len(rec.steps) == 1

    def test_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(target="y", feature=["x1", "x2"])
        fitted = rec.add(StepDropCorr(["x1", "x2"])).fit(_df())
        assert fitted.roles.of("y") == {"target"}

    def test_custom_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(feature=["x1", "x2"], group="y")
        fitted = rec.fit(_df())
        assert fitted.roles.of("y") == {"group"}

    def test_declared_features_suppress_auto_assignment(self) -> None:
        # An explicit feature assignment turns off the fit-time fallback, so
        # the unassigned column stays role-less.
        fitted = Recipe(data=_df()).set_roles(feature=["x1"]).fit(_df())
        assert fitted.roles.columns_with_role("feature") == ["x1"]
        assert fitted.roles.of("x2") == set()

    def test_target_excluded_from_auto_assigned_features(self) -> None:
        fitted = Recipe(data=_df()).set_roles(target="y").fit(_df())
        assert fitted.roles.of("y") == {"target"}
        assert fitted.roles.columns_with_role("feature") == ["x1", "x2"]

    def test_case_weights_excluded_from_auto_assigned_features(self) -> None:
        fitted = Recipe(data=_df()).set_roles(case_weights="x2").fit(_df())
        assert fitted.roles.columns_with_role("feature") == ["y", "x1"]

    def test_custom_role_only_column_still_auto_assigned_feature(self) -> None:
        # Only 'target' and 'case_weights' are reserved at fit time; a custom
        # role does not keep a column out of the feature fallback.
        fitted = Recipe(data=_df()).set_roles(group="x2").fit(_df())
        assert fitted.roles.of("x2") == {"group", "feature"}

    def test_selectors_see_assigned_roles(self) -> None:
        rec = (
            Recipe(data=_df())
            .set_roles(target="y", feature=["x1", "x2"])
            .add(StepNormalize(s.numeric_features()))
        )
        result = rec.fit(_df()).transform(_df())
        # 'y' is a target, so normalization left it alone.
        assert result["y"].tolist() == [1.0, 2.0, 3.0]
        assert result["x1"].mean() == pytest.approx(0.0)

    def test_repr_summarises_roles(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df()).set_roles(target="y", feature=["x1", "x2"])
        assert repr(rec) == snapshot

    def test_works_on_polars(self) -> None:
        df = pl.DataFrame(_df())
        rec = Recipe(data=df).set_roles(target="y", feature=["x1", "x2"])
        assert rec.roles.of("y") == {"target"}
        fitted = rec.fit(df)
        assert fitted.roles.columns_with_role("feature") == ["x1", "x2"]


class TestPositionalRoles:
    """add_role / update_role / remove_role take effect where they are written."""

    def test_role_change_affects_only_later_steps(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="id", old_role="feature")
            .add(StepNormalize(s.numeric_features()))
        )
        fitted = rec.fit(_df())
        assert fitted.roles.of("x2") == {"id"}
        # the second step saw only x1 (x2 had left the features)
        result = fitted.transform(_df())
        assert result["x2"].tolist() == [7.0, 8.0, 9.0]

    def test_earlier_step_sees_old_role(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(s.numeric_features()))
            .remove_role("x2", role="feature")
        )
        result = rec.fit(_df()).transform(_df())
        assert result["x2"].mean() == pytest.approx(0.0)

    def test_selector_lead_matches_set_roles(self) -> None:
        positional = (
            Recipe(data=_df()).add_role("x2", role="group").add_role("y", role="target")
        )
        declared = Recipe(data=_df()).set_roles(group="x2", target="y")
        assert positional.fit(_df()).roles == declared.fit(_df()).roles

    def test_leading_update_role_before_defaulting(self) -> None:
        fitted = (
            Recipe(data=_df()).update_role("x2", new_role="case_weights").fit(_df())
        )
        assert fitted.roles.of("x2") == {"case_weights"}
        assert fitted.roles.columns_with_role("feature") == ["y", "x1"]

    def test_add_role_on_produced_column(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .add_role(s.has_role("feature"), role="seen")
        )
        fitted = rec.fit(_df())
        assert fitted.roles.columns_with_role("seen") == ["x1", "x2"]

    def test_remove_role_and_lineage_refresh(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .remove_role("x2", role="feature")
        )
        fitted = rec.fit(_df())
        assert fitted.roles.of("x2") == set()
        summary = fitted.summary()
        assert summary["roles"].to_list()[summary["name"].to_list().index("x2")] == ""

    def test_update_without_old_role_replaces_all(self) -> None:
        rec = Recipe(data=_df(), target="y", features=["x1", "x2"]).set_roles(
            group="x1"
        )
        fitted = rec.update_role("x1", new_role="id").fit(_df())
        assert fitted.roles.of("x1") == {"id"}

    def test_update_with_old_role_keeps_others(self) -> None:
        rec = Recipe(data=_df(), target="y", features=["x1", "x2"]).set_roles(
            group="x1"
        )
        fitted = rec.update_role("x1", new_role="id", old_role="group").fit(_df())
        assert fitted.roles.of("x1") == {"feature", "id"}

    def test_pure(self) -> None:
        base = Recipe(data=_df())
        base.add_role("x1", role="group")
        assert base._role_ops == []  # pyright: ignore[reportPrivateUsage]

    def test_empty_selection_is_noop(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add_role(s.starts_with("zzz"), role="group")
            .fit(_df())
        )
        assert fitted.roles.role_names() == ["feature", "target"]

    def test_missing_role_raises(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df(), target="y").remove_role("x1", role="group")
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_df())
        assert str(excinfo.value) == snapshot

    def test_repr_interleaves(self, snapshot: SnapshotAssertion) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .add_role("x2", role="group")
            .update_role("x1", new_role="id", old_role="feature")
            .add(StepNormalize(["x2"]))
            .remove_role("x2", role="group")
        )
        assert repr(rec) == snapshot

    def test_survives_add_and_set_params(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role("x1", role="group")
        rec = rec.add(StepNormalize(["x2"]))
        assert rec.fit(_df()).roles.of("x1") == {"feature", "group"}

    def test_survives_set_params_and_clone(self) -> None:
        from sklearn.base import clone

        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x2"]))
            .add_role("x1", role="group")
        )
        for other in (rec.set_params(), clone(rec)):
            assert other._role_ops == rec._role_ops  # pyright: ignore[reportPrivateUsage]
            assert other.fit(_df()).roles.of("x1") == {"feature", "group"}

    def test_late_role_change_keeps_consumed_column_required(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x1", new_role="case_weights")
        )
        fitted = rec.fit(_df())
        assert fitted.schema.transform_required["x1"]
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(_df().drop(columns="x1"))
        assert str(excinfo.value) == snapshot


def _mixed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "x1": [4.0, 5.0, 6.0, 7.0],
            "x2": [7.0, 8.0, 9.0, 1.0],
            "g": ["a", "b", "a", "b"],
        }
    )


class TestPositionalRoleSelectors:
    """Every way of naming columns works, against the columns live at that point."""

    @pytest.mark.parametrize(
        "columns",
        [
            ["x1", "x2"],
            s.starts_with("x"),
            s.has_role("feature") & s.starts_with("x"),
            cs.starts_with("x"),
            ncs.matches("^x"),
            s.numeric_features() - "y",
        ],
        ids=["names", "mlf", "composite", "polars", "narwhals", "difference"],
    )
    def test_selector_forms(self, columns: object) -> None:
        fitted = (
            Recipe(data=_mixed(), target="y")
            .add(StepNormalize(["x1"]))
            .add_role(columns, role="tagged")
            .fit(_mixed())
        )
        assert fitted.roles.columns_with_role("tagged") == ["x1", "x2"]

    def test_provenance_selector_sees_live_lineage(self) -> None:
        rec = (
            Recipe(data=_mixed(), target="y")
            .add(StepDummy(s.nominal_features()))
            .add_role(s.has_source("dummy"), role="indicator")
        )
        fitted = rec.fit(_mixed())
        tagged = fitted.roles.columns_with_role("indicator")
        assert tagged and all(c.startswith("g") for c in tagged)
        summary = fitted.summary()
        roles = dict(zip(summary["name"].to_list(), summary["roles"].to_list()))
        assert all("indicator" in roles[c] for c in tagged)

    def test_unknown_column_raises_eagerly(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_mixed(), target="y").add_role("nope", role="group")
        assert str(excinfo.value) == snapshot

    def test_unknown_column_after_step_raises_at_fit(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_mixed(), target="y")
            .add(StepNormalize(s.numeric_features()))
            .add_role("nope", role="group")
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_mixed())
        assert str(excinfo.value) == snapshot

    def test_removed_column_cannot_be_targeted_later(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_mixed(), target="y")
            .add(StepDummy(s.nominal_features()))
            .add_role("g", role="group")
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_mixed())
        assert str(excinfo.value) == snapshot

    def test_works_on_polars(self) -> None:
        df = pl.DataFrame(_mixed())
        fitted = (
            Recipe(data=df, target="y")
            .add(StepNormalize(["x1"]))
            .add_role(s.starts_with("x"), role="tagged")
            .fit(df)
        )
        assert fitted.roles.columns_with_role("tagged") == ["x1", "x2"]


class TestPositionalRoleOrdering:
    def test_operations_at_one_position_run_in_order(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y", features=["x1", "x2"])
            .add(StepNormalize(["x1"]))
            .add_role("x1", role="a")
            .update_role("x1", new_role="b", old_role="a")
            .add_role("x1", role="c")
            .remove_role("x1", role="b")
            .fit(_df())
        )
        assert fitted.roles.of("x1") == {"feature", "c"}

    def test_trailing_operations_after_last_step(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .add_role("x1", role="late")
            .fit(_df())
        )
        assert fitted.roles.of("x1") == {"feature", "late"}

    def test_interleaved_operations_see_their_own_position(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add_role("x1", role="first")
            .add(StepNormalize(s.has_role("first")))
            .add_role("x2", role="first")
            .add(StepNormalize(s.has_role("first") - s.has_role("done")))
        )
        result = rec.fit(_df()).transform(_df())
        assert result["x1"].mean() == pytest.approx(0.0)
        assert result["x2"].mean() == pytest.approx(0.0)

    def test_positions_count_row_filters_and_empty_selections(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepFilter(lambda d: d["x1"] > 4.0))
            .add_role("x1", role="after_filter")
            .add(StepNormalize(s.starts_with("zzz")))
            .add_role("x2", role="after_empty")
            .add(StepNormalize(s.has_role("after_empty")))
        )
        fitted = rec.fit(_df())
        assert fitted.roles.of("x1") == {"feature", "after_filter"}
        result = fitted.transform(_df())
        # normalized with the statistics of the filtered rows (x2 = 8, 9); the
        # filter itself is skipped at transform
        assert result["x2"].mean() == pytest.approx(-(0.5**0.5))
        assert result["x1"].tolist() == [4.0, 5.0, 6.0]


class TestPositionalRoleLifecycle:
    def test_refit_is_independent_and_repeatable(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .add_role("x1", role="group")
        )
        first = rec.fit(_df())
        second = rec.fit(_df())
        assert first.roles == second.roles
        assert first.roles is not second.roles
        assert repr(rec) == repr(rec)

    def test_set_params_keeps_operations(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepDropCorr(s.numeric_features(), threshold=0.9))
            .add_role("y", role="group")
        )
        changed = rec.set_params(corr__threshold=0.5)
        assert changed.fit(_df()).roles.of("y") == {"target", "group"}

    def test_sklearn_clone_keeps_operations(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role("x1", role="group")
        assert clone(rec).fit(_df()).roles.of("x1") == {"feature", "group"}
        wrapped = clone(RecipeTransformer(rec))
        wrapped.fit(_df())
        assert wrapped.fitted_.roles.of("x1") == {"feature", "group"}

    def test_json_round_trip_keeps_final_roles(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="id", old_role="feature")
            .fit(_df())
        )

        restored = FittedRecipe.from_json(fitted.to_json())
        assert restored.roles == fitted.roles
        assert restored.summary().equals(fitted.summary())

    def test_leading_operation_changes_transform_requirement(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .update_role("x2", new_role="case_weights")
            .fit(_df())
        )
        assert fitted.schema is not None
        assert fitted.schema.transform_required["x2"] is False

    def test_later_operation_rederives_transform_required(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="case_weights", old_role="feature")
            .fit(_df())
        )
        assert fitted.schema is not None
        assert fitted.schema.roles.of("x2") == {"feature"}
        assert fitted.schema.transform_required["x2"] is False
        assert fitted.roles.of("x2") == {"case_weights"}

    def test_declared_targets_replay_leading_ops(self) -> None:
        rec = Recipe(data=_df(), target="y").remove_role("y", role="target")
        assert rec._declared_targets() == []  # pyright: ignore[reportPrivateUsage]
        later = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="target", old_role="feature")
        )
        assert later._declared_targets() == ["y"]  # pyright: ignore[reportPrivateUsage]

    @pytest.mark.parametrize("bad", ["", None])
    def test_invalid_role_name_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError):
            Recipe(data=_df(), target="y").add_role("x1", role=bad)  # type: ignore[arg-type]


class TestPositionalRoleErrors:
    def test_update_old_role_missing(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df(), target="y").update_role(
            "x1", new_role="id", old_role="group"
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_df())
        assert str(excinfo.value) == snapshot

    def test_incompatible_role_message(self, snapshot: SnapshotAssertion) -> None:
        rec = Recipe(data=_df(), target="y", features=["x1", "x2"]).add_role(
            "x1", role="target"
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_df())
        assert str(excinfo.value) == snapshot

    def test_selector_matching_some_without_role_raises(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add_role("x1", role="group")
            .remove_role(s.starts_with("x"), role="group")
        )
        with pytest.raises(ValueError) as excinfo:
            rec.fit(_df())
        assert str(excinfo.value) == snapshot


class TestLeadingProvenanceSelectors:
    """Operations before the first step already see the input-column lineage."""

    @pytest.mark.parametrize("make", [s.has_source, s.has_ancestor])
    def test_original_columns_selected(self, make: object) -> None:
        selector = make("original")  # type: ignore[operator]
        fitted = (
            Recipe(data=_df(), target="y")
            .add_role(selector, role="g")
            .add(StepNormalize(["x1"]))
            .fit(_df())
        )
        assert fitted.roles.columns_with_role("g") == ["y", "x1", "x2"]

    def test_leading_operation_without_steps(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y", features=["x1", "x2"])
            .add_role(s.has_source("original") & s.numeric_features(), role="g")
            .fit(_df())
        )
        assert fitted.roles.columns_with_role("g") == ["x1", "x2"]
        summary = fitted.summary()
        roles = dict(zip(summary["name"].to_list(), summary["roles"].to_list()))
        assert roles["x1"] == "feature, g"


class TestBackendParity:
    @pytest.mark.parametrize("backend", [pd.DataFrame, pl.DataFrame])
    def test_same_roles_and_values(self, backend: type) -> None:
        df = backend(_mixed().to_dict("list"))
        rec = (
            Recipe(data=df, target="y")
            .add(StepDummy(s.nominal_features()))
            .add_role(s.has_source("dummy"), role="indicator")
            .update_role("x2", new_role="id", old_role="feature")
            .add(StepNormalize(s.numeric_features() - s.has_role("indicator")))
        )
        fitted = rec.fit(df)
        assert fitted.roles.of("x2") == {"id"}
        assert fitted.roles.columns_with_role("indicator")
        assert isinstance(fitted.transform(df), backend)


_BASE = {"y": {"target"}, "x1": {"feature"}, "x2": {"feature"}}


def _expect(**changes: set[str]) -> dict[str, set[str]]:
    """The baseline with some columns' roles replaced; an empty set means no roles."""
    merged = {**_BASE, **changes}
    return {column: held for column, held in merged.items() if held}


@dataclass(frozen=True)
class _Scenario:
    id: str
    ops: Callable[[Recipe], Recipe]
    before: dict[str, set[str]] | type[Exception]
    after: dict[str, set[str]] | type[Exception] | None = None  # None: same as before
    features: list[str] | None = None  # None: the default features


SCENARIOS = [
    # A. single operations
    _Scenario(
        "A1 add a custom role",
        lambda r: r.add_role("x1", role="group"),
        _expect(x1={"feature", "group"}),
    ),
    _Scenario(
        "A2 add the role a column already has; the other features stay",
        lambda r: r.add_role("x1", role="feature"),
        _expect(),
    ),
    _Scenario(
        "A3 remove a role",
        lambda r: r.remove_role("x1", role="feature"),
        _expect(x1=set()),
    ),
    _Scenario(
        "A4 remove a role the column lacks",
        lambda r: r.remove_role("x1", role="group"),
        ValueError,
    ),
    _Scenario(
        "A5 swap one role, keep the rest",
        lambda r: r.add_role("x1", role="group").update_role(
            "x1", new_role="id", old_role="group"
        ),
        _expect(x1={"feature", "id"}),
    ),
    _Scenario(
        "A6 replace all roles",
        lambda r: r.add_role("x1", role="group").update_role("x1", new_role="id"),
        _expect(x1={"id"}),
    ),
    _Scenario(
        "A7 swap a role the column lacks",
        lambda r: r.update_role("x1", new_role="id", old_role="group"),
        ValueError,
    ),
    _Scenario(
        "A8 a selection matching nothing",
        lambda r: r.add_role(s.starts_with("zzz"), role="group"),
        _expect(),
    ),
    _Scenario(
        "A9 a role selector sees the default features",
        lambda r: r.add_role(s.numeric_features(), role="group"),
        _expect(x1={"feature", "group"}, x2={"feature", "group"}),
    ),
    _Scenario(
        "A10 feature role on the target column",
        lambda r: r.add_role("y", role="feature"),
        ValueError,
    ),
    # B. the feature role never changes which other columns are features
    _Scenario(
        "B1 a net no-op sequence",
        lambda r: r.add_role("x1", role="tmp").update_role(
            "x1", new_role="feature", old_role="tmp"
        ),
        _expect(),
    ),
    _Scenario(
        "B2 a round trip through another role",
        lambda r: r.update_role("x1", new_role="id").update_role(
            "x1", new_role="feature", old_role="id"
        ),
        _expect(),
    ),
    _Scenario(
        "B3 removing every feature is not defaulted again",
        lambda r: r.remove_role("x1", role="feature").remove_role("x2", role="feature"),
        _expect(x1=set(), x2=set()),
    ),
    # C. target and case weights on a feature
    _Scenario(
        "C1 target on a default feature",
        lambda r: r.add_role("x1", role="target"),
        _expect(x1={"target"}),
    ),
    _Scenario(
        "C2 case weights on a default feature",
        lambda r: r.add_role("x1", role="case_weights"),
        _expect(x1={"case_weights"}),
    ),
    _Scenario(
        "C3 target on a declared feature",
        lambda r: r.add_role("x1", role="target"),
        ValueError,
        features=["x1", "x2"],
    ),
    _Scenario(
        "C4 case weights on a declared feature",
        lambda r: r.add_role("x1", role="case_weights"),
        _expect(x1={"feature", "case_weights"}),
        features=["x1", "x2"],
    ),
    _Scenario(
        "C5 a feature role given on purpose is not displaced later",
        lambda r: r.add_role("x1", role="feature").add_role("x1", role="target"),
        ValueError,
    ),
    _Scenario(
        "C6 swap feature for target",
        lambda r: r.update_role("x1", new_role="target", old_role="feature"),
        _expect(x1={"target"}),
    ),
    _Scenario(
        "C7 replace all roles with target",
        lambda r: r.update_role("x1", new_role="target"),
        _expect(x1={"target"}),
    ),
    _Scenario(
        "C8 update, then claim the column",
        lambda r: r.update_role("x1", new_role="id", old_role="feature").add_role(
            "x1", role="target"
        ),
        _expect(x1={"id", "target"}),
    ),
    _Scenario(
        "C9 remove the feature role, then case weights",
        lambda r: r.remove_role("x1", role="feature").add_role(
            "x1", role="case_weights"
        ),
        _expect(x1={"case_weights"}),
    ),
]


class TestRoleOperationScenarios:
    """Each row is one rule, run before the first step and after a step.

    A row states one expected result that must hold at both positions unless it
    also gives an ``after`` result, so the rows that differ are exactly where
    position is allowed to matter. Messages are pinned elsewhere.
    """

    @pytest.mark.parametrize("where", ["before", "after"])
    @pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda sc: sc.id)
    def test_scenario(self, scenario: _Scenario, where: str) -> None:
        rec = Recipe(data=_df(), target="y", features=scenario.features)
        if where == "after":
            rec = rec.add(StepNormalize(["x2"]))
        rec = scenario.ops(rec)

        expected = scenario.before
        if where == "after" and scenario.after is not None:
            expected = scenario.after
        if isinstance(expected, type):
            with pytest.raises(expected):
                rec.fit(_df())
        else:
            assert rec.fit(_df()).roles == Roles(expected)


class TestRoleOpsReflected:
    """Role operations show up in the untrained views and the sklearn wrapper."""

    def test_summary_reflects_leading_ops(self) -> None:
        rec = Recipe(data=_df(), target="y").update_role(
            "x2", new_role="id", old_role="feature"
        )
        summary = rec.summary()
        shown = dict(zip(summary["name"].to_list(), summary["roles"].to_list()))
        assert shown["x2"] == "id"

    def test_repr_inputs_reflect_leading_ops(self) -> None:
        plain = Recipe(data=_df(), target="y")
        changed = plain.update_role("x2", new_role="id", old_role="feature")
        assert repr(changed).splitlines()[1] != repr(plain).splitlines()[1]

    def test_leading_feature_op_without_declared_features(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y").remove_role("x1", role="feature").fit(_df())
        )
        assert fitted.roles.of("x1") == set()
        assert fitted.roles.columns_with_role("feature") == ["x2"]

    def test_leading_target_add_with_feature_op(self) -> None:
        fitted = (
            Recipe(data=_df())
            .add_role("y", role="target")
            .remove_role("x1", role="feature")
            .fit(_df())
        )
        assert fitted.roles.of("y") == {"target"}
        assert fitted.roles.columns_with_role("feature") == ["x2"]

    def test_leading_case_weights_add_with_feature_op(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add_role("x2", role="case_weights")
            .remove_role("x1", role="feature")
            .fit(_df())
        )
        assert fitted.roles.of("x2") == {"case_weights"}

    def test_leading_update_role_without_old_role_replaces_all(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y").update_role("x2", new_role="id").fit(_df())
        )
        assert fitted.roles.of("x2") == {"id"}
        assert fitted.roles.of("x1") == {"feature"}

    def test_recipe_transformer_selector_target(self) -> None:
        df = _df()
        rec = Recipe(data=df).add_role(s.starts_with("y"), role="target")
        transformer = RecipeTransformer(rec).fit(df[["x1", "x2"]], df["y"])
        out = transformer.transform(df[["x1", "x2"]])
        assert list(out.columns) == ["x1", "x2"]

    def test_recipe_transformer_late_target_does_not_reject_earlier_step(
        self,
    ) -> None:
        df = _df()
        rec = (
            Recipe(data=df, target="y")
            .add(StepNormalize(["x2"]))
            .update_role("x2", new_role="target", old_role="feature")
        )
        RecipeTransformer(rec).fit(df[["y", "x1", "x2"]])

    def test_summary_leading_provenance_selector(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role(
            s.has_source("original"), role="g"
        )
        shown = dict(
            zip(rec.summary()["name"].to_list(), rec.summary()["roles"].to_list())
        )
        assert "g" in shown["x1"]

    def test_recipe_transformer_add_role_target(self) -> None:
        df = _df()
        rec = Recipe(data=df).add_role("y", role="target")
        transformer = RecipeTransformer(rec).fit(df[["x1", "x2"]], df["y"])
        out = transformer.transform(df[["x1", "x2"]])
        assert list(out.columns) == ["x1", "x2"]

    def test_recipe_transformer_leading_feature_add_keeps_every_feature(self) -> None:
        # Adding the feature role to x1 must not take it from x2: both are
        # features, so a step on every numeric feature normalizes both.
        df = _df()
        rec = (
            Recipe(data=df, target="y")
            .add_role("x1", role="feature")
            .add(StepNormalize(s.numeric_features()))
        )
        features = df[["x1", "x2"]]
        out = RecipeTransformer(rec).fit(features, df["y"]).transform(features)
        assert list(out.columns) == ["x1", "x2"]
        assert out["x1"].mean() == pytest.approx(0.0)
        assert out["x2"].mean() == pytest.approx(0.0)

    def test_recipe_transformer_late_target_is_dropped(self) -> None:
        df = _df()
        rec = (
            Recipe(data=df, target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="target", old_role="feature")
        )
        # x2 only becomes a target after the step, so it is a feature at
        # transform time; the wrapper strips whatever the fitted roles call a
        # target.
        transformer = RecipeTransformer(rec).fit(df[["y", "x1", "x2"]])
        out = transformer.transform(df[["y", "x1", "x2"]])
        assert "x2" not in out.columns


def _shown(recipe: Recipe) -> dict[str, str]:
    summary = recipe.summary()
    return dict(zip(summary["name"].to_list(), summary["roles"].to_list()))


class TestLeadingOpsReadSequentially:
    """Operations before the first step read like the same ones after a step."""

    def test_target_or_case_weights_displace_only_the_default_role(self) -> None:
        # Declared features are the user's own choice and are never displaced:
        # the same operation on a declared feature is the conflict it looks like.
        declared = Recipe(data=_df(), target="y", features=["x1", "x2"]).add_role(
            "x1", role="case_weights"
        )
        assert declared.fit(_df()).roles.of("x1") == {"feature", "case_weights"}
        default = Recipe(data=_df(), target="y").add_role("x1", role="case_weights")
        assert default.fit(_df()).roles.of("x1") == {"case_weights"}

    def test_set_roles_declares_the_features(self) -> None:
        # A declaration chooses the features, so no other default survives it,
        # also when a leading operation sits in front of it.
        plain = Recipe(data=_df(), target="y").set_roles(feature="x1")
        after_op = (
            Recipe(data=_df(), target="y")
            .add_role("x2", role="g")
            .set_roles(feature="x1")
        )
        assert plain.fit(_df()).roles.columns_with_role("feature") == ["x1"]
        assert after_op.fit(_df()).roles.columns_with_role("feature") == ["x1"]
        assert after_op.fit(_df()).roles.of("x2") == {"g"}

    def test_declared_features_are_not_defaulted_again(self) -> None:
        # update_role without old_role drops the feature role of x1; x2 must
        # not be promoted to a feature because of it.
        leading = Recipe(data=_df(), target="y", features=["x1"]).update_role(
            "x1", new_role="id"
        )
        after_step = (
            Recipe(data=_df(), target="y", features=["x1"])
            .add(StepNormalize(["x1"]))
            .update_role("x1", new_role="id")
        )
        for rec in (leading, after_step):
            fitted = rec.fit(_df())
            assert fitted.roles.columns_with_role("feature") == []
            assert fitted.roles.of("x2") == set()

    def test_role_selector_works_on_polars(self) -> None:
        df = pl.DataFrame(_df())
        rec = Recipe(data=df, target="y").remove_role(
            s.numeric_features() & s.starts_with("x"), role="feature"
        )
        assert rec.fit(df).roles.columns_with_role("feature") == []

    def test_untrained_views_show_only_declared_roles(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role(s.numeric_features(), role="g")
        assert _shown(rec) == {"y": "target", "x1": "g", "x2": "g"}
        assert _shown(Recipe(data=_df(), target="y")) == {
            "y": "target",
            "x1": "",
            "x2": "",
        }
        fitted = rec.fit(_df())
        summary = fitted.summary()
        shown = dict(zip(summary["name"].to_list(), summary["roles"].to_list()))
        assert shown["x1"] == "feature, g"

    def test_untrained_views_show_features_declared_by_set_roles(self) -> None:
        # Folded into the declared roles: no other default feature is left.
        rec = Recipe(data=_df(), target="y").set_roles(feature="x1")
        assert _shown(rec) == {"y": "target", "x1": "feature", "x2": ""}

    def test_set_roles_before_a_leading_operation(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .set_roles(group="x1")
            .update_role("x1", new_role="id")
        )
        assert rec.fit(_df()).roles.of("x1") == {"id"}


class TestRoleSelectorOrder:
    """A role selector returns columns in the data frame's order."""

    def test_a_round_trip_through_a_role_change_keeps_the_order(self) -> None:
        # x1 loses its only role and gets one back, which moves it to the end
        # of the roles' own bookkeeping; the selection must not follow.
        rec = (
            Recipe(data=_df(), target="y")
            .update_role("x1", new_role="id")
            .update_role("x1", new_role="feature", old_role="id")
            .add(StepNormalize(s.numeric_features()))
        )
        fitted = rec.fit(_df())
        assert list(fitted.steps[0].schema_in().columns) == ["x1", "x2"]

    def test_declared_order_does_not_reorder_the_selection(self) -> None:
        rec = Recipe(data=_df(), target="y", features=["x2", "x1"]).add(
            StepNormalize(s.numeric_features())
        )
        fitted = rec.fit(_df())
        assert list(fitted.steps[0].schema_in().columns) == ["x1", "x2"]

    def test_a_custom_role_is_selected_in_data_frame_order(self) -> None:
        # x2 was given the role first, so the roles hold it first.
        rec = (
            Recipe(data=_df(), target="y")
            .add_role("x2", role="group")
            .add_role("x1", role="group")
            .add(StepNormalize(s.has_role("group")))
        )
        fitted = rec.fit(_df())
        assert list(fitted.steps[0].schema_in().columns) == ["x1", "x2"]

    def test_works_on_polars(self) -> None:
        df = pl.DataFrame(_df())
        rec = (
            Recipe(data=df, target="y")
            .update_role("x1", new_role="id")
            .update_role("x1", new_role="feature", old_role="id")
            .add(StepNormalize(s.numeric_features()))
        )
        fitted = rec.fit(df)
        assert list(fitted.steps[0].schema_in().columns) == ["x1", "x2"]


class TestSetRolesKeepsWrittenOrder:
    """``set_roles()`` acts on the input columns, in order with leading operations."""

    def test_set_roles_after_a_leading_operation(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .update_role("x1", new_role="id")
            .set_roles(group="x1")
        )
        assert rec.fit(_df()).roles.of("x1") == {"id", "group"}

    def test_roles_property_reflects_set_roles_after_a_leading_operation(self) -> None:
        rec = Recipe(data=_df()).add_role("x1", role="id").set_roles(target="y")
        assert rec.roles.columns_with_role("target") == ["y"]
        assert "id" in rec.roles.of("x1")

    def test_set_roles_after_the_steps_still_acts_on_the_input(self) -> None:
        rec = (
            Recipe(data=_df())
            .add_role("x2", role="g")
            .add(StepNormalize(["x1"]))
            .set_roles(target="y")
        )
        fitted = rec.fit(_df())
        assert fitted.roles.of("y") == {"target"}
        assert fitted.roles.of("x2") == {"feature", "g"}

    def test_set_roles_order_across_several_calls(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add_role("x1", role="a")
            .set_roles(b="x1")
            .remove_role("x1", role="a")
            .set_roles(a="x1")
        )
        assert rec.fit(_df()).roles.of("x1") == {"feature", "a", "b"}

    def test_set_roles_target_after_a_leading_operation(self) -> None:
        rec = (
            Recipe(data=_df())
            .add_role("x2", role="g")
            .set_roles(target="y", case_weights="x1")
        )
        fitted = rec.fit(_df())
        assert fitted.roles.of("y") == {"target"}
        assert fitted.roles.of("x1") == {"case_weights"}
        assert fitted.roles.columns_with_role("feature") == ["x2"]

    def test_repr_shows_the_call_that_was_written(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add_role("x2", role="group")
            .set_roles(group=["x1", "x2"], case_weights="x1")
        )
        assert repr(rec) == snapshot

    def test_unknown_column_is_rejected_eagerly(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = Recipe(data=_df(), target="y").add_role("x2", role="g")
        with pytest.raises(ValueError) as excinfo:
            rec.set_roles(group="nope")
        assert str(excinfo.value) == snapshot

    def test_plural_features_is_still_rejected(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role("x2", role="g")
        with pytest.raises(ValueError):
            rec.set_roles(features="x1")

    def test_receiver_is_unchanged(self) -> None:
        base = Recipe(data=_df(), target="y").add_role("x2", role="g")
        base.set_roles(group="x1")
        assert len(base._role_ops) == 1  # pyright: ignore[reportPrivateUsage]
        assert base.fit(_df()).roles.of("x1") == {"feature"}

    def test_declared_targets_see_set_roles_after_an_operation(self) -> None:
        rec = Recipe().add_role("x2", role="g").set_roles(target="y")
        assert rec._declared_targets() == ["y"]  # pyright: ignore[reportPrivateUsage]

    def test_incompatible_roles_are_rejected_at_the_call(self) -> None:
        # The same call raises at once on a recipe with no leading operation.
        with pytest.raises(ValueError):
            Recipe(data=_df()).set_roles(target="x1", feature="x1")
        rec = Recipe(data=_df(), target="y").add_role("x2", role="g")
        with pytest.raises(ValueError):
            rec.set_roles(target="x1", feature="x1")

    def test_incompatible_with_the_roles_the_operations_left(self) -> None:
        declared = Recipe(data=_df(), target="y", features=["x1"]).add_role(
            "x2", role="g"
        )
        with pytest.raises(ValueError):
            declared.set_roles(target="x1")
        # Once an earlier operation took the feature role away, it is allowed.
        freed = (
            Recipe(data=_df(), target="y", features=["x1"])
            .remove_role("x1", role="feature")
            .set_roles(target="x1")
        )
        assert freed.fit(_df()).roles.of("x1") == {"target"}

    def test_an_empty_role_name_is_rejected_either_way(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as plain:
            Recipe(data=_df()).set_roles(**{"": "x1"})
        with pytest.raises(ValueError) as leading:
            Recipe(data=_df()).add_role("x2", role="g").set_roles(**{"": "x1"})
        assert str(plain.value) == snapshot
        assert str(leading.value) == str(plain.value)

    def test_a_bad_earlier_operation_is_reported_by_fit_not_by_set_roles(
        self,
    ) -> None:
        rec = Recipe(data=_df(), target="y").remove_role("x1", role="group")
        rec = rec.set_roles(group="x2")
        with pytest.raises(ValueError):
            rec.fit(_df())

    def test_without_prototype_one_call_still_checks_itself(self) -> None:
        rec = Recipe().add_role("x2", role="g")
        with pytest.raises(ValueError):
            rec.set_roles(target="x1", feature="x1")

    def test_without_prototype_it_checks_against_what_is_declared(self) -> None:
        constructed = Recipe(target="y", features=["x1"]).add_role("x2", role="g")
        with pytest.raises(ValueError):
            constructed.set_roles(target="x1")
        with pytest.raises(ValueError):
            constructed.set_roles(feature="y")

    def test_without_prototype_it_checks_against_earlier_plain_name_ops(self) -> None:
        rec = Recipe(target="y").add_role("x1", role="feature")
        with pytest.raises(ValueError):
            rec.set_roles(target="x1")

    def test_without_prototype_any_column_name_is_accepted(self) -> None:
        # There is nothing to check names against, so a typo only surfaces
        # when the recipe is fitted, as it does without a leading operation.
        rec = Recipe(target="y").add_role("x2", role="g").set_roles(group="typo")
        with pytest.raises(ValueError):
            rec.fit(_df())

    def test_roles_attribute_includes_later_set_roles_assignments(self) -> None:
        rec = Recipe(data=_df(), target="y").add_role("x2", role="g")
        later = rec.set_roles(group="x1")
        assert later.roles.of("x1") == {"group"}
        assert _shown(later)["x1"] == "group"


def _with_weights() -> pd.DataFrame:
    return _df().assign(w=[0.5, 1.0, 2.0])


class TestLateRoleOpsAndRequirements:
    """What `transform()` needs after a role operation written after a step."""

    def test_late_role_gets_the_requirement_override_of_that_role(self) -> None:
        df = _df()
        rec = (
            Recipe(data=df, target="y", features=["x1"])
            .update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x1"]))
            .add_role("x2", role="id")
        )
        fitted = rec.fit(df)
        assert fitted.schema.transform_required["x2"] is False
        assert "x2" not in fitted.transform(df.drop(columns="x2")).columns

    def test_override_matches_the_same_operation_before_the_step(self) -> None:
        df = _df()
        early = (
            Recipe(data=df, target="y", features=["x1"])
            .update_role_requirements("id", transform_required=False)
            .add_role("x2", role="id")
            .add(StepNormalize(["x1"]))
        )
        late = (
            Recipe(data=df, target="y", features=["x1"])
            .update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x1"]))
            .add_role("x2", role="id")
        )
        assert (
            early.fit(df).schema.transform_required
            == late.fit(df).schema.transform_required
        )

    def test_selector_override_applies_to_a_late_role(self) -> None:
        df = _df()
        rec = (
            Recipe(data=df, target="y", features=["x1"])
            .update_role_requirements(s.has_role("id"), transform_required=False)
            .add(StepNormalize(["x1"]))
            .add_role("x2", role="id")
        )
        assert rec.fit(df).schema.transform_required["x2"] is False

    def test_row_filter_does_not_make_every_column_required(self) -> None:
        df = _with_weights()
        rec = (
            Recipe(data=df, target="y")
            .add(StepFilter(lambda d: d["x1"] > 4.0))
            .add(StepNormalize(["x1"]))
            .update_role("w", new_role="case_weights", old_role="feature")
        )
        fitted = rec.fit(df)
        assert fitted.schema.transform_required["w"] is False
        assert "w" not in fitted.transform(df.drop(columns="w")).columns

    def test_step_without_a_selector_only_reads_what_it_declares(self) -> None:
        df = _with_weights()
        rec = (
            Recipe(data=df, target="y")
            .add(StepIntercept())
            .add(StepNormalize(["x1"]))
            .update_role("w", new_role="case_weights", old_role="feature")
        )
        assert rec.fit(df).schema.transform_required["w"] is False

    def test_column_read_by_a_step_stays_required(self) -> None:
        df = _with_weights()
        rec = (
            Recipe(data=df, target="y")
            .add(StepNormalize(["w"]))
            .update_role("w", new_role="case_weights", old_role="feature")
        )
        assert rec.fit(df).schema.transform_required["w"] is True

    def test_target_turned_into_a_feature_and_read_stays_required(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("y", new_role="feature", old_role="target")
            .add(StepNormalize(["y"]))
        )
        fitted = rec.fit(_df())
        assert fitted.schema.transform_required["y"] is True
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(_df().drop(columns="y"))
        assert str(excinfo.value) == snapshot

    def test_target_turned_into_a_feature_is_required_even_if_unread(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("y", new_role="feature", old_role="target")
        )
        assert rec.fit(_df()).schema.transform_required["y"] is True

    def test_untouched_columns_keep_their_requirement(self) -> None:
        df = _with_weights()
        rec = (
            Recipe(data=df, target="y", case_weights="w")
            .add(StepNormalize(["x1"]))
            .add_role("x2", role="g")
        )
        required = rec.fit(df).schema.transform_required
        assert required == {"y": False, "x1": True, "x2": True, "w": False}

    def test_roles_recorded_in_the_schema_are_those_of_the_input(self) -> None:
        fitted = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="case_weights", old_role="feature")
            .fit(_df())
        )
        assert fitted.schema.roles.of("x2") == {"feature"}
        assert fitted.schema.transform_required["x2"] is False

    def test_fit_does_not_share_a_schema_between_fits(self) -> None:
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x1"]))
            .update_role("x2", new_role="case_weights", old_role="feature")
        )
        first, second = rec.fit(_df()), rec.fit(_df())
        assert first.schema is not second.schema
        assert first.schema.transform_required == second.schema.transform_required


class TestRoleOpArguments:
    def test_list_of_names_is_copied(self) -> None:
        names = ["x1"]
        rec = (
            Recipe(data=_df(), target="y")
            .add(StepNormalize(["x2"]))
            .add_role(names, role="g")
        )
        names.append("x2")
        op = rec._role_ops[0]  # pyright: ignore[reportPrivateUsage]
        assert op.plain_names() == ["x1"]
        assert rec.fit(_df()).roles.columns_with_role("g") == ["x1"]

    def test_update_role_names_its_own_parameter(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).update_role("x1", new_role="")
        assert str(excinfo.value) == snapshot

    def test_old_role_is_named_when_invalid(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).update_role("x1", new_role="id", old_role="")
        assert str(excinfo.value) == snapshot

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("The column was not found", "the column was not found"),
            ("StepFoo could not run.", "StepFoo could not run"),
            ("ID clash", "ID clash"),
            # A backend's own name reads like a sentence start but is a name.
            ("Polars could not find the column", "Polars could not find the column"),
            ("Narwhals: unknown column", "Narwhals: unknown column"),
            (
                "Pandas says no.\nThe column was not found",
                "Pandas says no; the column was not found",
            ),
            # Only a single trailing period goes, so an ellipsis survives.
            ("The columns were truncated...", "the columns were truncated..."),
            (
                "Polars failed\nThe second line.",
                "Polars failed; the second line",
            ),
            (
                "The columns were not found: ['a']\n\nHint: Did you mean ['b']",
                "the columns were not found: ['a']; did you mean ['b']",
            ),
        ],
    )
    def test_wrapped_error_keeps_every_line_and_identifiers(
        self, text: str, expected: str
    ) -> None:
        from mlfeatures.recipe import _lower  # pyright: ignore[reportPrivateUsage]

        assert _lower(ValueError(text)) == expected
