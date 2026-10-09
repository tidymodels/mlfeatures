"""Instance attributes are declared at class level (issue #169).

A class-body annotation documents an attribute and lets the type-checkers see
it, but creates nothing at runtime: ``__init__`` must still assign every
instance attribute it declares. These tests pin that both ways, so a
declaration cannot drift from what the constructor actually sets.
"""

from __future__ import annotations

import inspect
from typing import Any

import pandas as pd
import pytest

import mlfeatures
from mlfeatures import (
    ColSchema,
    Lineage,
    Recipe,
    Roles,
    SchemaInfo,
    StepNormalize,
)
from mlfeatures.compat import RecipeTransformer
from mlfeatures.step import RowOp, Step


def _declared(cls: type) -> set[str]:
    """Names annotated in ``cls``'s own body, excluding ``ClassVar`` members."""
    return {
        name
        for name, annotation in cls.__dict__.get("__annotations__", {}).items()
        if "ClassVar" not in str(annotation)
    }


def _concrete_steps() -> list[type[Step]]:
    steps: list[type[Step]] = []
    for name in mlfeatures.__all__:
        obj = getattr(mlfeatures, name)
        if inspect.isclass(obj) and issubclass(obj, Step) and obj not in (Step, RowOp):
            steps.append(obj)
    return steps


def _declaring_class(cls: type, attr: str) -> type | None:
    """The class nearest ``cls`` in its MRO that annotates ``attr``."""
    for klass in cls.__mro__:
        if attr in klass.__dict__.get("__annotations__", {}):
            return klass
    return None


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Roles(), id="Roles"),
        pytest.param(lambda: Lineage(), id="Lineage"),
        pytest.param(lambda: Recipe(), id="Recipe"),
        pytest.param(lambda: SchemaInfo(), id="SchemaInfo"),
        pytest.param(lambda: ColSchema(), id="ColSchema"),
    ],
)
def test_every_declared_attribute_is_set_by_init(build: Any) -> None:
    """A bare annotation creates no attribute, so ``__init__`` must assign each one."""
    instance = build()
    declared = _declared(type(instance))
    assert declared, f"{type(instance).__name__} declares no attributes at class level"
    missing = sorted(name for name in declared if not hasattr(instance, name))
    assert not missing, f"declared but never assigned in __init__: {missing}"


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: Roles(), id="Roles"),
        pytest.param(lambda: Lineage(), id="Lineage"),
        pytest.param(lambda: Recipe(), id="Recipe"),
        pytest.param(lambda: SchemaInfo(), id="SchemaInfo"),
        pytest.param(lambda: ColSchema(), id="ColSchema"),
    ],
)
def test_every_assigned_attribute_is_declared(build: Any) -> None:
    """The reverse direction: nothing ``__init__`` sets is missing a declaration."""
    instance = build()
    undeclared = sorted(set(vars(instance)) - _declared(type(instance)))
    assert not undeclared, f"assigned in __init__ but not declared: {undeclared}"


def test_declaring_attributes_does_not_share_state_between_instances() -> None:
    """Annotations carry no value, so two instances never share a container."""
    first, second = Roles(), Roles()
    first.add("a", "feature")
    assert second.of("a") == set()
    assert Lineage()._live is not Lineage()._live  # pyright: ignore[reportPrivateUsage]
    assert Recipe().steps is not Recipe().steps


class TestRecipeTransformerFitState:
    """The fit-time attributes are declared but exist only once ``fit()`` has run."""

    FIT_ATTRS = ("fitted_", "feature_names_in_", "feature_names_out_")

    def test_declared_on_the_class(self) -> None:
        assert set(self.FIT_ATTRS) <= _declared(RecipeTransformer)

    def test_absent_before_fit(self) -> None:
        """sklearn's ``hasattr``-based fitted checks depend on this."""
        transformer = RecipeTransformer(Recipe().add(StepNormalize(["a"])))
        for attr in self.FIT_ATTRS:
            assert not hasattr(transformer, attr)

    def test_present_after_fit(self) -> None:
        transformer = RecipeTransformer(Recipe().add(StepNormalize(["a"])))
        transformer.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
        for attr in self.FIT_ATTRS:
            assert hasattr(transformer, attr)


class TestStepFitStateDeclarations:
    """Fit-state attributes live on ``Step``; each step narrows ``_estimates``."""

    SHARED = ("_estimates", "_schema_in", "_schema_out", "_output_parents")

    def test_shared_attributes_are_declared_on_step(self) -> None:
        assert set(self.SHARED) <= _declared(Step)

    @pytest.mark.parametrize("step_cls", _concrete_steps(), ids=lambda c: c.__name__)
    def test_step_initializes_every_shared_attribute(
        self, step_cls: type[Step]
    ) -> None:
        """No step re-declares ``_schema_in`` / ``_schema_out`` / ``_output_parents``.

        They are owned by ``Step`` and set to ``None`` in ``Step.__init__``; a
        step that re-declares one has gone back to the pre-hoisting pattern.
        """
        for attr in ("_schema_in", "_schema_out", "_output_parents"):
            assert _declaring_class(step_cls, attr) is Step, (
                f"{step_cls.__name__} re-declares {attr!r}; it is declared once on Step"
            )

    @pytest.mark.parametrize("step_cls", _concrete_steps(), ids=lambda c: c.__name__)
    def test_estimating_step_narrows_estimates(self, step_cls: type[Step]) -> None:
        """A step that stores estimates declares their type below ``Step``.

        ``Step`` types ``_estimates`` as ``Any``, so without a narrowing
        annotation the type-checkers would see nothing about what ``fit()``
        stores. Steps that estimate nothing (they only record schemas) are
        exempt: their ``_estimates`` stays ``None`` for good.
        """
        source = inspect.getsource(step_cls)
        stores_estimates = "self._estimates =" in source
        if not stores_estimates and _declaring_class(step_cls, "_estimates") is Step:
            pytest.skip("step stores no fit-time estimates")
        assert _declaring_class(step_cls, "_estimates") is not Step, (
            f"{step_cls.__name__} stores _estimates but does not narrow its "
            "type with a class-level annotation"
        )
