"""Contract-surface tests for the ``Step`` / ``RowOp`` ABCs.

These check only that the contract is importable and behaves as an abstract base
class. The full per-step conformance suite is a separate issue (#17).
"""

from abc import ABC

import pytest

from shoez import ANY_DTYPE, ColSchema, Mode, RowOp, SchemaInfo, Step
from shoez.types import _AnyDtype


class _ConcreteStep(Step):
    """Minimal concrete step used to verify the contract can be implemented."""

    def fit(self, training_data: object, info: object) -> "Step":
        self.trained = True
        return self

    def transform(self, new_data: object) -> object:
        return new_data

    def schema_in(self) -> ColSchema:
        return ColSchema()

    def schema_out(self) -> ColSchema:
        return ColSchema()

    def accepts_dtypes(self) -> "set[str] | _AnyDtype":
        return ANY_DTYPE

    @property
    def produces_sparse(self) -> bool:
        return False


class _ConcreteRowOp(RowOp):
    def fit(self, training_data: object, info: object) -> "Step":
        return self

    def transform(self, new_data: object) -> object:
        return new_data

    def schema_in(self) -> ColSchema:
        return ColSchema()

    def schema_out(self) -> ColSchema:
        return ColSchema()

    def accepts_dtypes(self) -> "set[str] | _AnyDtype":
        return {"int"}

    @property
    def produces_sparse(self) -> bool:
        return False


def test_step_is_abc() -> None:
    assert issubclass(Step, ABC)


def test_rowop_subclasses_step() -> None:
    assert issubclass(RowOp, Step)


def test_step_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Step()  # type: ignore[abstract]


def test_rowop_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        RowOp()  # type: ignore[abstract]


def test_concrete_step_can_be_instantiated() -> None:
    step = _ConcreteStep()
    assert step.fit(None, None) is step
    assert step.transform("data") == "data"
    assert isinstance(step.schema_in(), ColSchema)
    assert step.produces_sparse is False


def test_concrete_rowop_can_be_instantiated() -> None:
    op = _ConcreteRowOp()
    assert isinstance(op, Step)
    assert op.accepts_dtypes() == {"int"}


def test_required_pkgs_defaults_to_empty() -> None:
    assert _ConcreteStep().required_pkgs() == []


def test_any_dtype_contains_everything() -> None:
    assert "float" in ANY_DTYPE
    assert "anything-at-all" in ANY_DTYPE


def test_any_dtype_is_singleton() -> None:
    assert _AnyDtype() is ANY_DTYPE


def test_schema_info_importable() -> None:
    assert isinstance(SchemaInfo(), SchemaInfo)


def test_mode_has_seven_members() -> None:
    assert {m.name for m in Mode} == {
        "ONE_TO_ONE",
        "ONE_TO_MANY",
        "MANY_TO_MANY",
        "REMOVING",
        "ADDING",
        "ROW_OP",
        "UNCONSTRAINED",
    }
