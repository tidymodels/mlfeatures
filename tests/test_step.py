"""Contract-surface tests for the ``Step`` / ``RowOp`` ABCs.

These check only that the contract is importable and behaves as an abstract base
class. The full per-step conformance suite is a separate issue (#17).
"""

from abc import ABC

import narwhals.stable.v2 as nw
import pytest

from shoez import ANY_DTYPE, ColSchema, Mode, RowOp, SchemaInfo, Step
from shoez._dependencies import MissingDependencyError
from shoez.types import AcceptedDtypes, _AnyDtype


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

    def accepts_dtypes(self) -> AcceptedDtypes:
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

    def accepts_dtypes(self) -> AcceptedDtypes:
        return {nw.Int64}

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
    assert op.accepts_dtypes() == {nw.Int64}


def test_required_pkgs_defaults_to_empty() -> None:
    assert _ConcreteStep().required_pkgs() == []


class _MissingPkgStep(_ConcreteStep):
    """A step declaring a package that is guaranteed not to be installed."""

    id = "needs_zzz"

    def required_pkgs(self) -> list[str]:
        return ["shoez_no_such_dependency_zzz"]


def test_check_required_pkgs_passes_when_nothing_is_declared() -> None:
    _ConcreteStep().check_required_pkgs()


def test_check_required_pkgs_reports_the_step_and_the_package() -> None:
    with pytest.raises(MissingDependencyError) as excinfo:
        _MissingPkgStep().check_required_pkgs()
    message = str(excinfo.value)
    assert "shoez_no_such_dependency_zzz" in message
    assert "needs_zzz" in message
    assert "pip install" in message


class _InstalledPkgStep(_ConcreteStep):
    """A step declaring an installed package, with a version constraint."""

    id = "needs_numpy"

    def required_pkgs(self) -> list[str]:
        return ["numpy>=1.0"]


def test_check_required_pkgs_passes_for_an_installed_declaration() -> None:
    _InstalledPkgStep().check_required_pkgs()


def test_any_dtype_contains_everything() -> None:
    assert nw.Float64 in ANY_DTYPE
    assert nw.Categorical in ANY_DTYPE


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
