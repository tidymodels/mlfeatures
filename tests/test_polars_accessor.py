"""Tests for the opt-in ``shz`` polars accessor (``shoez.polars``).

The accessor is thin sugar over :class:`~shoez.recipe.Recipe`: every method must
produce exactly what the equivalent one-step recipe produces, and always return a
polars ``DataFrame``. Importing :mod:`shoez.polars` registers the namespace as a
side effect.
"""

from __future__ import annotations

import inspect

import polars as pl
import pytest

import shoez
import shoez.polars  # noqa: F401  (registers the `shz` namespace)
from shoez.polars import ShzNamespace
from shoez.recipe import Recipe
from shoez.step import RowOp, Step, _step_type_label
from shoez.step_class_distance import StepClassDistance
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_impute_mean import StepImputeMean
from shoez.step_impute_median import StepImputeMedian
from shoez.step_impute_mode import StepImputeMode
from shoez.step_intercept import StepIntercept
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot
from shoez.step_polynomial import StepPolynomial
from shoez.step_unknown import StepUnknown


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
        assert hasattr(df, "shz")

    def test_returns_polars_dataframe(self, df: pl.DataFrame) -> None:
        assert isinstance(df.shz.step_dummy(["color"]), pl.DataFrame)

    def test_does_not_mutate_source(self, df: pl.DataFrame) -> None:
        before = df.clone()
        df.shz.step_dummy(["color"])
        assert df.equals(before)


# ---------------------------------------------------------------------------
# Each wrapper maps to the right step
# ---------------------------------------------------------------------------


# Each case is (accessor method name, step class, shared args). We call the
# wrapper and the constructor with the same args and assert they agree — i.e. the
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
    got = getattr(df.shz, method)(*args)
    assert got.equals(_recipe_out(df, step_cls(*args)))


# ---------------------------------------------------------------------------
# Generic apply & chaining
# ---------------------------------------------------------------------------


class TestApplyAndChaining:
    def test_apply_single_step(self, df: pl.DataFrame) -> None:
        assert df.shz.apply(StepDummy(["color"])).equals(df.shz.step_dummy(["color"]))

    def test_apply_multiple_steps_matches_recipe(self, df: pl.DataFrame) -> None:
        steps = [StepDummy(["color"]), StepNormalize(["x"])]
        assert df.shz.apply(*steps).equals(_recipe_out(df, *steps))

    def test_apply_no_steps_is_identity(self, df: pl.DataFrame) -> None:
        assert df.shz.apply().equals(df)

    def test_chaining_matches_multi_step_recipe(self, df: pl.DataFrame) -> None:
        chained = df.shz.step_dummy(["color"]).shz.step_normalize(["x"])
        assert chained.equals(
            _recipe_out(df, StepDummy(["color"]), StepNormalize(["x"]))
        )


# ---------------------------------------------------------------------------
# Wrapper/step parity — the accessor must not silently fall out of sync
# ---------------------------------------------------------------------------


def _public_step_classes() -> list[type[Step]]:
    """Every public, concrete `Step` exported from ``shoez.__all__``.

    The abstract bases (`Step`, `RowOp`) are excluded; everything else that is a
    `Step` subclass is a step that end users can build, so it must be reachable
    through the ``df.shz`` accessor with a dedicated ``step_*`` wrapper.
    """
    out: list[type[Step]] = []
    for name in shoez.__all__:
        obj = getattr(shoez, name)
        if inspect.isclass(obj) and issubclass(obj, Step) and obj not in (Step, RowOp):
            out.append(obj)
    return out


class TestWrapperStepParity:
    """Guards the invariant behind issue #73: the accessor stays in sync.

    Rather than hand-checking that ``shoez/polars.py`` was updated whenever a step
    is added, these tests fail in CI the moment a public step has no matching
    wrapper (or a wrapper points at a step that no longer exists).
    """

    def test_every_public_step_has_a_wrapper(self) -> None:
        missing = [
            step_cls.__name__
            for step_cls in _public_step_classes()
            if not hasattr(ShzNamespace, _step_type_label(step_cls))
        ]
        assert not missing, (
            f"public step(s) with no `df.shz` wrapper: {missing}; "
            "add a matching `step_*` method to `ShzNamespace` in shoez/polars.py"
        )

    def test_no_wrapper_without_a_step(self) -> None:
        wrapper_names = {
            name
            for name in vars(ShzNamespace)
            if name.startswith("step_") and callable(getattr(ShzNamespace, name))
        }
        step_names = {_step_type_label(cls) for cls in _public_step_classes()}
        orphaned = wrapper_names - step_names
        assert not orphaned, (
            f"`ShzNamespace` wrapper(s) with no matching public step: {orphaned}"
        )
