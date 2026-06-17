"""Roles + canonical constructor (set_roles, formula sugar).

Covers role assignment on a recipe and the minimal formula parser, both of
which are pure (return a new recipe) and validate against the prototype data.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

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

    def test_equality(self) -> None:
        assert Roles({"a": {"predictor"}}) == Roles({"a": {"predictor"}})
        assert Roles({"a": {"predictor"}}) != Roles({"a": {"outcome"}})

    def test_equality_with_non_roles(self) -> None:
        assert Roles() != "not a Roles"


@pytest.mark.skip(reason="set_roles is not yet implemented")
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

    def test_unknown_column_rejected(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(data=_df()).set_roles(outcome="missing")
        assert str(excinfo.value) == snapshot

    def test_without_prototype_skips_existence_check(self) -> None:
        rec = Recipe().set_roles(outcome="anything")
        assert rec.roles.of("anything") == {"outcome"}

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

    def test_roles_survive_add(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y").add(StepDropCorr(["x1", "x2"]))
        assert rec.roles.of("y") == {"outcome"}
        assert len(rec.steps) == 1

    def test_roles_survive_fit(self) -> None:
        rec = Recipe(data=_df()).set_roles(outcome="y", predictor=["x1", "x2"])
        fitted = rec.add(StepDropCorr(["x1", "x2"])).fit(_df())
        assert fitted.roles.of("y") == {"outcome"}
