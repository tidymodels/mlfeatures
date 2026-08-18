"""Does a step defined outside the ``shoez`` package still work everywhere?

`tests._plugin_fixture.StepScaleShift` stands in for a third-party step. These
tests push it through every surface a plugin uses -- `Recipe`, the conformance
contract, serialization, the polars accessor, the sklearn adapter -- so that
breaking the extensibility path fails shoez's own CI rather than being noticed
later by someone building against the package.

What this can and cannot cover: a fixture living in ``tests/`` shares an
interpreter and a source tree with shoez, so it cannot detect problems that only
appear across a distribution boundary -- packaging, editable installs,
type-checker resolution, or real entry-point discovery. `TestPublicApiOnly` is
what keeps the rest honest: without it the fixture could quietly reach into
``shoez._frame`` and still pass.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import polars as pl
import pytest

import shoez
from shoez import Recipe, RecipeTransformer, from_json, minimize, to_json
from shoez import Selectors as s
from shoez.testing import StepConformanceTest
from tests._plugin_fixture import StepScaleShift

DATA = {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "g": ["x", "y", "x", "y"]}

#: Imports the fixture is allowed to make from outside ``shoez``'s public API.
#: Every entry is a gap a plugin author hits too, so each names the issue that
#: tracks it. Shrinking this to ``{}`` is the goal; adding to it should be a
#: deliberate decision made in review, which is the point of pinning it here.
PRIVATE_IMPORT_ALLOWLIST = {
    # `Selector` annotates every step's `columns`; `AcceptedDtypes` is the
    # declared return of the abstract `accepts_dtypes()`. Neither is exported.
    "shoez.types": {"Selector", "AcceptedDtypes"},  # #197
}


def frame() -> pd.DataFrame:
    return pd.DataFrame(DATA)


class TestPublicApiOnly:
    """The fixture must be writable against ``shoez.__all__``.

    This is the test that gives the others meaning. It parses the fixture rather
    than importing it, so it sees what the module *asks for*, and fails on any
    new reach into a private module.
    """

    @staticmethod
    def _shoez_imports() -> list[tuple[str, str]]:
        """Every ``(module, name)`` the fixture imports from shoez."""
        source = pathlib.Path(__file__).parent / "_plugin_fixture.py"
        tree = ast.parse(source.read_text())
        found: list[tuple[str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] == "shoez":
                    found.extend((node.module, alias.name) for alias in node.names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "shoez":
                        found.append((alias.name, ""))
        return found

    def test_fixture_imports_only_public_names_or_allowlisted_ones(self) -> None:
        offenders = [
            f"{module}.{name}"
            for module, name in self._shoez_imports()
            if name not in PRIVATE_IMPORT_ALLOWLIST.get(module, set())
            and not (module == "shoez" and name in shoez.__all__)
        ]
        assert not offenders, (
            f"the plugin fixture reached outside shoez's public API: {offenders}. "
            "Either export the name from `shoez.__all__`, or add it to "
            "PRIVATE_IMPORT_ALLOWLIST with the issue that tracks it."
        )

    def test_allowlist_is_still_needed(self) -> None:
        """Fails once an allowlisted name becomes public, so the list cannot rot."""
        stale = [
            f"{module}.{name}"
            for module, names in PRIVATE_IMPORT_ALLOWLIST.items()
            for name in names
            if name in shoez.__all__
        ]
        assert not stale, (
            f"now public and should be imported from `shoez` directly: {stale}; "
            "remove from PRIVATE_IMPORT_ALLOWLIST and update the fixture"
        )

    def test_fixture_lives_outside_the_shoez_package(self) -> None:
        """A fixture inside `shoez/` would prove nothing about extensibility."""
        assert not StepScaleShift.__module__.startswith("shoez.")


class TestConformance(StepConformanceTest):
    """The shipped step contract, run against an out-of-package step."""

    training_data = DATA

    def make_step(self) -> StepScaleShift:
        return StepScaleShift(["a", "b"])


class TestRecipeIntegration:
    def test_fit_and_transform(self) -> None:
        df = frame()
        fitted = Recipe().add(StepScaleShift(s.numeric_predictors())).fit(df)
        assert list(fitted.transform(df)["a"]) == pytest.approx([-1.5, -0.5, 0.5, 1.5])

    def test_composes_with_a_first_party_step(self) -> None:
        from shoez import StepNormalize

        df = frame()
        fitted = Recipe().add(StepNormalize(["b"])).add(StepScaleShift(["a"])).fit(df)
        out = fitted.transform(df)
        assert list(out["a"]) == pytest.approx([-1.5, -0.5, 0.5, 1.5])
        assert sum(out["b"]) == pytest.approx(0.0)

    def test_hyperparameter_is_honoured(self) -> None:
        df = frame()
        fitted = Recipe().add(StepScaleShift(["a"], shift=10.0)).fit(df)
        assert list(fitted.transform(df)["a"]) == pytest.approx([8.5, 9.5, 10.5, 11.5])

    def test_empty_selection_is_the_identity(self) -> None:
        df = frame()
        # No date columns in the frame, so this selector resolves to nothing.
        fitted = Recipe().add(StepScaleShift(s.date_predictors())).fit(df)
        assert fitted.transform(df).equals(df)
        assert fitted.steps[0]._empty_selection is True

    def test_repr_derives_the_snake_case_label(self) -> None:
        recipe = Recipe().add(StepScaleShift(["a"]))
        assert "step_scale_shift" in repr(recipe)
        assert "[untrained]" in repr(recipe)
        assert "[trained]" in repr(recipe.fit(frame()))


class TestSerialization:
    def test_round_trip_by_qualified_name(self) -> None:
        df = frame()
        fitted = Recipe().add(StepScaleShift(["a", "b"], shift=5.0)).fit(df)

        payload = to_json(fitted)
        assert "tests._plugin_fixture.StepScaleShift" in payload

        reloaded = from_json(payload)
        assert reloaded.steps[0].shift == 5.0
        assert reloaded.transform(df).equals(fitted.transform(df))

    def test_minimize_preserves_behaviour(self) -> None:
        df = frame()
        fitted = Recipe().add(StepScaleShift(["a"])).fit(df)
        assert minimize(fitted).transform(df).equals(fitted.transform(df))


class TestPolarsAccessor:
    def test_apply_is_the_plugin_route(self) -> None:
        import shoez.polars  # noqa: F401  (registers the namespace)

        out = pl.DataFrame(DATA).shz.apply(StepScaleShift(["a"]))
        assert out["a"].to_list() == pytest.approx([-1.5, -0.5, 0.5, 1.5])

    def test_out_of_package_step_does_not_break_wrapper_parity(self) -> None:
        """Importing a plugin step must not make shoez's own parity test fail."""
        from shoez.polars import ShzNamespace

        wrappers = {n for n in vars(ShzNamespace) if n.startswith("step_")}
        assert "step_scale_shift" not in wrappers


class TestSklearnCompat:
    def test_fit_transform_and_feature_names(self) -> None:
        df = frame()
        y = pd.Series([0, 0, 1, 1])
        recipe = Recipe().set_roles(outcome="target").add(StepScaleShift(["a", "b"]))

        transformer = RecipeTransformer(recipe)
        transformer.fit(df, y)
        assert list(transformer.get_feature_names_out()) == list(df.columns)

    def test_pipeline_round_trip(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        df = frame()
        y = pd.Series([0, 0, 1, 1])
        recipe = Recipe().set_roles(outcome="target").add(StepScaleShift(["a", "b"]))
        pipe = Pipeline(
            [("prep", RecipeTransformer(recipe)), ("model", LogisticRegression())]
        )
        pipe.fit(df.drop(columns=["g"]), y)
        assert len(pipe.predict(df.drop(columns=["g"]))) == 4
