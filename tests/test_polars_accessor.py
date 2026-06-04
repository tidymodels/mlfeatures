"""Tests for the opt-in ``shz`` polars accessor (``shoez.polars``).

The accessor is thin sugar over :class:`~shoez.recipe.Recipe`: every method must
produce exactly what the equivalent one-step recipe produces, and always return a
polars ``DataFrame``. Importing :mod:`shoez.polars` registers the namespace as a
side effect.
"""

from __future__ import annotations

import polars as pl
import pytest

import shoez.polars  # noqa: F401  (registers the `shz` namespace)
from shoez.recipe import Recipe
from shoez.step import Step
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_intercept import StepIntercept
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot


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
