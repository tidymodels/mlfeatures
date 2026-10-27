"""Tests for the opt-in ``mlf`` polars accessor (``mlfeatures.polars``).

The accessor is thin sugar over :class:`~mlfeatures.recipe.Recipe`: every method must
produce exactly what the equivalent one-step recipe produces, and always return a
polars ``DataFrame``. Importing :mod:`mlfeatures.polars` registers the namespace as a
side effect.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import polars as pl
import pytest

import mlfeatures
import mlfeatures.polars  # noqa: F401  (registers the `mlf` namespace)
from mlfeatures.polars import MlfNamespace
from mlfeatures.recipe import Recipe
from mlfeatures.step import RowOp, Step, _step_type_label
from mlfeatures.step_class_distance import StepClassDistance
from mlfeatures.step_drop_corr import StepDropCorr
from mlfeatures.step_drop_zero_variance import StepDropZeroVariance
from mlfeatures.step_dummy import StepDummy
from mlfeatures.step_impute_mean import StepImputeMean
from mlfeatures.step_impute_median import StepImputeMedian
from mlfeatures.step_impute_mode import StepImputeMode
from mlfeatures.step_intercept import StepIntercept
from mlfeatures.step_log import StepLog
from mlfeatures.step_normalize import StepNormalize
from mlfeatures.step_one_hot import StepOneHot
from mlfeatures.step_polynomial import StepPolynomial
from mlfeatures.step_range import StepRange
from mlfeatures.step_unknown import StepUnknown


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "color": ["red", "blue", "red", "green", "blue"],
            "size": ["s", "m", "l", "m", "s"],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x2": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated with x
            "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        }
    )


def _recipe_out(df: pl.DataFrame, *steps: Step) -> pl.DataFrame:
    """The reference result: a one-shot recipe fitted and applied to ``df``."""
    out = Recipe(steps=list(steps)).fit(df).transform(df)
    assert isinstance(out, pl.DataFrame)
    return out


# ---------------------------------------------------------------------------
# Registration & return type
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_namespace_registered(self, df: pl.DataFrame) -> None:
        assert hasattr(df, "mlf")

    def test_returns_polars_dataframe(self, df: pl.DataFrame) -> None:
        assert isinstance(df.mlf.step_dummy(["color"]), pl.DataFrame)

    def test_does_not_mutate_source(self, df: pl.DataFrame) -> None:
        before = df.clone()
        df.mlf.step_dummy(["color"])
        assert df.equals(before)


# ---------------------------------------------------------------------------
# Each wrapper maps to the right step
# ---------------------------------------------------------------------------


# Each case is (accessor method name, step class, shared args). We call the
# wrapper and the constructor with the same args and assert they agree, i.e. the
# wrapper forwards to the step it claims to. Step *behaviour* is covered by each
# step's own test suite (and the equivalence machinery itself by the apply tests).
@pytest.mark.parametrize(
    ("method", "step_cls", "args"),
    [
        ("step_dummy", StepDummy, (["color"],)),
        ("step_one_hot", StepOneHot, (["color"],)),
        ("step_drop_corr", StepDropCorr, (["x", "x2"],)),
        ("step_drop_zero_variance", StepDropZeroVariance, (["x", "const"],)),
        ("step_intercept", StepIntercept, ()),
        ("step_normalize", StepNormalize, (["x"],)),
        ("step_range", StepRange, (["x"],)),
        ("step_log", StepLog, (["x"],)),
        ("step_class_distance", StepClassDistance, (["x", "x2"], "color")),
        ("step_impute_mean", StepImputeMean, (["x"],)),
        ("step_impute_median", StepImputeMedian, (["x"],)),
        ("step_impute_mode", StepImputeMode, (["color"],)),
        ("step_polynomial", StepPolynomial, (["x"],)),
        ("step_unknown", StepUnknown, (["color"],)),
    ],
)
def test_wrapper_maps_to_step(
    df: pl.DataFrame,
    method: str,
    step_cls: type[Step],
    args: tuple[object, ...],
) -> None:
    got = getattr(df.mlf, method)(*args)
    assert got.equals(_recipe_out(df, step_cls(*args)))


# ---------------------------------------------------------------------------
# Generic apply & chaining
# ---------------------------------------------------------------------------


class TestApplyAndChaining:
    def test_apply_single_step(self, df: pl.DataFrame) -> None:
        assert df.mlf.apply(StepDummy(["color"])).equals(df.mlf.step_dummy(["color"]))

    def test_apply_multiple_steps_matches_recipe(self, df: pl.DataFrame) -> None:
        steps = [StepDummy(["color"]), StepNormalize(["x"])]
        assert df.mlf.apply(*steps).equals(_recipe_out(df, *steps))

    def test_apply_no_steps_is_identity(self, df: pl.DataFrame) -> None:
        assert df.mlf.apply().equals(df)

    def test_chaining_matches_multi_step_recipe(self, df: pl.DataFrame) -> None:
        chained = df.mlf.step_dummy(["color"]).mlf.step_normalize(["x"])
        assert chained.equals(
            _recipe_out(df, StepDummy(["color"]), StepNormalize(["x"]))
        )


# ---------------------------------------------------------------------------
# Wrapper/step parity: the accessor must not silently fall out of sync
# ---------------------------------------------------------------------------


def _public_step_classes() -> list[type[Step]]:
    """Every public, concrete `Step` exported from ``mlfeatures.__all__``.

    The abstract bases (`Step`, `RowOp`) are excluded; everything else that is a
    `Step` subclass is a step that end users can build, so it must be reachable
    through the ``df.mlf`` accessor with a dedicated ``step_*`` wrapper.
    """
    out: list[type[Step]] = []
    for name in mlfeatures.__all__:
        obj = getattr(mlfeatures, name)
        if inspect.isclass(obj) and issubclass(obj, Step) and obj not in (Step, RowOp):
            out.append(obj)
    return out


class TestWrapperStepParity:
    """Guards the invariant behind issue #73: the accessor stays in sync.

    Rather than hand-checking that ``mlfeatures/polars.py`` was updated whenever a step
    is added, these tests fail in CI the moment a public step has no matching
    wrapper (or a wrapper points at a step that no longer exists).
    """

    def test_every_public_step_has_a_wrapper(self) -> None:
        missing = [
            step_cls.__name__
            for step_cls in _public_step_classes()
            if not hasattr(MlfNamespace, _step_type_label(step_cls))
        ]
        assert not missing, (
            f"public step(s) with no `df.mlf` wrapper: {missing}; "
            "add a matching `step_*` method to `MlfNamespace` in mlfeatures/polars.py"
        )

    def test_no_wrapper_without_a_step(self) -> None:
        wrapper_names = {
            name
            for name in vars(MlfNamespace)
            if name.startswith("step_") and callable(getattr(MlfNamespace, name))
        }
        step_names = {_step_type_label(cls) for cls in _public_step_classes()}
        orphaned = wrapper_names - step_names
        assert not orphaned, (
            f"`MlfNamespace` wrapper(s) with no matching public step: {orphaned}"
        )

    #: Per-step type aliases that are private to their module (not in
    #: `mlfeatures.__all__`), so a wrapper cannot reference them by name and instead
    #: spells out the expansion. Keyed by the alias name a step's annotation may
    #: use, mapped to the literal expansion a wrapper is expected to write instead.
    _PRIVATE_ALIAS_EXPANSIONS: ClassVar[dict[str, str]] = {
        "Predicate": "Callable[[Frame], Any]",
        "Mapping[str, Mutation]": "Mapping[str, Callable[[Frame], Any]]",
    }

    def test_wrapper_signature_mirrors_step_constructor(self) -> None:
        """A wrapper's parameters must match the step's, name/kind/default/type alike.

        The wrapper is allowed to drop the step's `id` parameter (the accessor
        discards the fitted step, so a caller-chosen handle is moot), and is
        allowed to spell out a step-private type alias (`_PRIVATE_ALIAS_EXPANSIONS`)
        since it cannot import a name that isn't in `mlfeatures.__all__`. Nothing else
        may differ: every other parameter must appear in the wrapper with the same
        default and the same annotation. This catches both a wrapper silently
        missing an entire argument and a wrapper typing `columns` narrower than
        the step actually accepts (e.g. `list[str]` instead of `Selector`).
        """
        mismatches: list[str] = []
        for step_cls in _public_step_classes():
            label = _step_type_label(step_cls)
            wrapper = getattr(MlfNamespace, label, None)
            if wrapper is None:
                continue  # reported by test_every_public_step_has_a_wrapper
            step_params = inspect.signature(step_cls.__init__).parameters
            wrapper_params = inspect.signature(wrapper).parameters
            for name, step_param in step_params.items():
                if name in ("self", "id"):
                    continue
                wrapper_param = wrapper_params.get(name)
                if wrapper_param is None:
                    mismatches.append(f"{label}: wrapper is missing parameter {name!r}")
                    continue
                step_annotation = self._PRIVATE_ALIAS_EXPANSIONS.get(
                    step_param.annotation, step_param.annotation
                )
                if wrapper_param.annotation != step_annotation:
                    mismatches.append(
                        f"{label}.{name}: wrapper annotation "
                        f"{wrapper_param.annotation!r} != step annotation "
                        f"{step_param.annotation!r}"
                    )
                if wrapper_param.default != step_param.default:
                    mismatches.append(
                        f"{label}.{name}: wrapper default {wrapper_param.default!r} "
                        f"!= step default {step_param.default!r}"
                    )
            extra = set(wrapper_params) - set(step_params) - {"self"}
            if extra:
                mismatches.append(f"{label}: wrapper has extra parameter(s) {extra}")
        assert not mismatches, "\n".join(mismatches)
