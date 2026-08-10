"""Tests for where a recipe checks a step's ``required_pkgs()``.

Optional dependencies are only needed to *run* a step, so the check happens at
``Recipe.fit()`` and ``FittedRecipe.transform()``. Building, inspecting, and
serializing a recipe must stay possible in an environment that lacks them. See
``design/extensibility.md``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shoez import ANY_DTYPE, ColSchema, Recipe, Step
from shoez._dependencies import MissingDependencyError
from shoez.types import AcceptedDtypes


class _MissingPkgStep(Step):
    """A step declaring a package that is guaranteed not to be installed."""

    id = "needs_zzz"
    # Real steps set these in their own ``__init__``; this one has none.
    trained = False
    _empty_selection = False

    def required_pkgs(self) -> list[str]:
        return ["shoez_no_such_dependency_zzz"]

    def fit(self, training_data: object, info: object) -> "Step":
        self.trained = True
        return self

    def transform(self, new_data: object) -> object:
        return new_data

    def schema_in(self) -> ColSchema:
        return ColSchema()

    def schema_out(self) -> ColSchema:
        return ColSchema()

    def accepts_dtypes(self) -> AcceptedDtypes:
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        return False


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1.0, 2.0]})


def test_add_does_not_check_required_pkgs() -> None:
    recipe = Recipe().add(_MissingPkgStep())
    assert len(recipe.steps) == 1


def test_inspecting_a_recipe_does_not_check_required_pkgs(df: pd.DataFrame) -> None:
    recipe = Recipe(data=df).add(_MissingPkgStep())
    assert "untrained" in repr(recipe)
    recipe.summary()


def test_fit_checks_required_pkgs(df: pd.DataFrame) -> None:
    recipe = Recipe().add(_MissingPkgStep())
    with pytest.raises(MissingDependencyError, match="shoez_no_such_dependency_zzz"):
        recipe.fit(df)


def test_fit_checks_steps_passed_to_the_constructor(df: pd.DataFrame) -> None:
    # ``Recipe(steps=[...])`` bypasses ``add()``, so ``fit()`` is the only gate.
    recipe = Recipe(steps=[_MissingPkgStep()])
    with pytest.raises(MissingDependencyError):
        recipe.fit(df)


class _TogglePkgStep(_MissingPkgStep):
    """Declares the missing package only while ``declaring`` is set."""

    id = "toggle_zzz"
    declaring = False

    def required_pkgs(self) -> list[str]:
        return ["shoez_no_such_dependency_zzz"] if _TogglePkgStep.declaring else []


def test_transform_checks_required_pkgs(df: pd.DataFrame) -> None:
    # Fit with the package "available", then transform without it, standing in
    # for a recipe deserialized into a leaner environment.
    fitted = Recipe().add(_TogglePkgStep()).fit(df)
    _TogglePkgStep.declaring = True
    try:
        with pytest.raises(
            MissingDependencyError, match="shoez_no_such_dependency_zzz"
        ):
            fitted.transform(df)
    finally:
        _TogglePkgStep.declaring = False
