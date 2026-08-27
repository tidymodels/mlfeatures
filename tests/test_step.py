"""Contract-surface tests for the ``Step`` / ``RowOp`` ABCs.

These check only that the contract is importable and behaves as an abstract base
class. The full per-step conformance suite is a separate issue (#17).
"""

from abc import ABC

import narwhals.stable.v2 as nw
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import (
    ANY_DTYPE,
    ColSchema,
    KeepOriginals,
    Mode,
    RowOp,
    SchemaInfo,
    Step,
    StepDummy,
)
from shoez._dependencies import MissingDependencyError
from shoez.types import AcceptedDtypes, _AnyDtype


class _ConcreteStep(Step):
    """Minimal concrete step used to verify the contract can be implemented."""

    mode = Mode.ONE_TO_ONE
    default_id = "concrete"

    def __init__(self, id: str | None = None) -> None:
        super().__init__(id, self.default_id)

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
    def __init__(self) -> None:
        super().__init__(None, "concrete_rowop")

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

    default_id = "needs_zzz"

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

    default_id = "needs_numpy"

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


class TestKeepOriginalColsFollowsMode:
    """The value is derived from `Mode` and cannot be changed afterwards (#176)."""

    def test_every_mode_has_a_policy(self) -> None:
        assert {KeepOriginals.for_mode(m) for m in Mode} == set(KeepOriginals)

    @pytest.mark.parametrize(
        ("policy", "expected"),
        [
            (KeepOriginals.KEEP, True),
            (KeepOriginals.DISCARD, False),
            (KeepOriginals.CONFIGURABLE, None),
        ],
    )
    def test_fixed_value(self, policy: KeepOriginals, expected: bool | None) -> None:
        assert policy.fixed_value() is expected

    def test_discarding_mode_gets_false_without_saying_so(self) -> None:
        assert _ConcreteStep().keep_original_cols is False

    def test_keeping_mode_gets_true_without_saying_so(self) -> None:
        # ``_ConcreteRowOp`` passes no ``keep_original_cols``; ROW_OP fixes it.
        assert _ConcreteRowOp().keep_original_cols is True

    def test_configurable_mode_takes_the_caller_s_value(self) -> None:
        assert StepDummy(["x"]).keep_original_cols is False
        assert StepDummy(["x"], keep_original_cols=True).keep_original_cols is True

    def test_contradicting_a_discarding_mode_raises(
        self, snapshot: SnapshotAssertion
    ) -> None:
        class _BadStep(_ConcreteStep):
            def __init__(self) -> None:
                super(_ConcreteStep, self).__init__(
                    None, "bad", keep_original_cols=True
                )

        with pytest.raises(ValueError) as excinfo:
            _BadStep()
        assert str(excinfo.value) == snapshot

    def test_contradicting_a_keeping_mode_raises(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # The mirror of the case above: ``False`` is not the argument's default
        # any more, so stating it against a keeping mode is caught too.
        class _BadRowOp(_ConcreteRowOp):
            def __init__(self) -> None:
                super(_ConcreteRowOp, self).__init__(
                    None, "bad", keep_original_cols=False
                )

        with pytest.raises(ValueError) as excinfo:
            _BadRowOp()
        assert str(excinfo.value) == snapshot

    def test_reassignment_is_rejected(self) -> None:
        step = _ConcreteStep()
        with pytest.raises(AttributeError):
            step.keep_original_cols = True  # type: ignore[misc]

    def test_rowop_supplies_its_own_mode(self) -> None:
        assert RowOp.mode is Mode.ROW_OP

    def test_a_step_without_a_mode_cannot_be_constructed(
        self, snapshot: SnapshotAssertion
    ) -> None:
        class _ModelessStep(_ConcreteStep):
            mode = None  # type: ignore[assignment]

        with pytest.raises(TypeError) as excinfo:
            _ModelessStep()
        assert str(excinfo.value) == snapshot


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
