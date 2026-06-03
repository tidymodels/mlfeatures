"""Roles + canonical constructor (set_roles, formula sugar).

Covers role assignment on a recipe and the minimal formula parser, both of
which are pure (return a new recipe) and validate against the prototype data.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, Roles, StepDropCorr


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

    def test_incompatible_combo_rejected(self) -> None:
        roles = Roles()
        roles.add("a", "outcome")
        with pytest.raises(ValueError, match="incompatible"):
            roles.add("a", "predictor")

    def test_copy_is_independent(self) -> None:
        roles = Roles({"a": {"predictor"}})
        clone = roles.copy()
        clone.add("a", "group")
        assert roles.of("a") == {"predictor"}

    def test_equality(self) -> None:
        assert Roles({"a": {"predictor"}}) == Roles({"a": {"predictor"}})
        assert Roles({"a": {"predictor"}}) != Roles({"a": {"outcome"}})

    def test_equality_with_non_roles(self) -> None:
        assert Roles() != "not a Roles"


class TestSetRoles:
    def test_assigns_standard_roles(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        assert rec.roles.of("y") == {"outcome"}
        assert rec.roles.columns_with_role("predictor") == ["x1", "x2"]

    def test_custom_role(self) -> None:
        rec = Recipe(data=_df()).set_roles(predictor=["x1"], group=["x2"])
        assert rec.roles.of("x2") == {"group"}

    def test_is_pure(self) -> None:
        base = Recipe(data=_df())
        extended = base.set_roles(outcome="y")
        assert base.roles.of("y") == set()
        assert extended.roles.of("y") == {"outcome"}
        assert base is not extended

    def test_unknown_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="not in the prototype"):
            Recipe(data=_df()).set_roles(outcome="missing")

    def test_without_prototype_skips_existence_check(self) -> None:
        rec = Recipe().set_roles(outcome="anything")
        assert rec.roles.of("anything") == {"outcome"}

    def test_roles_survive_add(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y").add(StepDropCorr(["x1", "x2"]))
        assert rec.roles.of("y") == {"outcome"}
        assert len(rec.steps) == 1

    def test_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        fitted = rec.add(StepDropCorr(["x1", "x2"])).fit(_df())
        assert fitted.roles.of("y") == {"outcome"}


class TestFormula:
    def test_dot_rhs_makes_all_others_predictors(self) -> None:
        rec = Recipe(data=_df(), formula="y ~ .")
        assert rec.roles.of("y") == {"outcome"}
        assert set(rec.roles.columns_with_role("predictor")) == {"x1", "x2"}

    def test_explicit_predictors(self) -> None:
        rec = Recipe(data=_df(), formula="y ~ x1 + x2")
        assert set(rec.roles.columns_with_role("predictor")) == {"x1", "x2"}
        assert rec.roles.of("y") == {"outcome"}

    def test_unlisted_column_gets_no_role(self) -> None:
        # x3 is in the data but absent from the formula: it stays roleless,
        # retained in the schema but not selected as a predictor or outcome.
        df = _df().assign(x3=[10.0, 11.0, 12.0])
        rec = Recipe(data=df, formula="y ~ x1 + x2")
        assert rec.roles.of("x3") == set()
        assert "x3" not in rec.roles.columns_with_role("predictor")

    def test_polars_prototype(self) -> None:
        rec = Recipe(data=pl.DataFrame(_df()), formula="y ~ .")
        assert set(rec.roles.columns_with_role("predictor")) == {"x1", "x2"}

    def test_requires_data(self) -> None:
        with pytest.raises(ValueError, match="requires a prototype"):
            Recipe(formula="y ~ .")

    def test_unknown_column_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown column"):
            Recipe(data=_df(), formula="y ~ nope")

    @pytest.mark.parametrize(
        "formula",
        ["y ~ x1 * x2", "y ~ x1:x2", "y ~ log(x1)", "y ~ x1 - x2", "y ~ x1^2"],
    )
    def test_unsupported_syntax_rejected(self, formula: str) -> None:
        with pytest.raises(ValueError, match="unsupported formula syntax"):
            Recipe(data=_df(), formula=formula)

    def test_requires_single_tilde(self) -> None:
        with pytest.raises(ValueError, match="exactly one"):
            Recipe(data=_df(), formula="y ~ x1 ~ x2")

    def test_empty_side_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            Recipe(data=_df(), formula="y ~ ")

    def test_malformed_term_rejected(self) -> None:
        with pytest.raises(ValueError, match="malformed"):
            Recipe(data=_df(), formula="y ~ x1 + ")
