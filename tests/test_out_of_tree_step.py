"""Hold the public API to its promise: a plugin can be written from `shoez` alone.

Issue #197 was found by hand, by writing an out-of-tree step and discovering it
could not be annotated without importing `shoez.types`. This module turns that
one-off probe into a standing check, so the gap cannot silently reopen the next
time a step's signature grows a new alias.

Three things are asserted here, and a fourth elsewhere:

1. The fixture step imports nothing private (``test_imports_only_public_names``).
2. It satisfies the full step contract (``TestDemoMaxAbsConformance``).
3. It works through a real `Recipe`, which is what a user would do with it.
4. It type-checks under ``mypy --strict`` and ``pyright --strict`` — enforced by
   CI running both over ``extras/``, not by a test here. Shelling out to a type
   checker from pytest would make the suite slow and network-dependent for a
   guarantee the lint job already gives us on every push.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest
from shoez_demo_step import StepDemoMaxAbs

from shoez import Recipe
from shoez import Selectors as s
from shoez.testing import StepConformanceTest

FIXTURE_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "extras"
    / "shoez_demo_step"
    / "__init__.py"
)

# The only shoez modules a plugin is allowed to import. `shoez.testing` is
# sanctioned because it is how a plugin runs the conformance contract; anything
# else (`shoez.types`, `shoez._frame`) means the public API was insufficient.
_ALLOWED_MODULES = {"shoez", "shoez.testing"}


def _imported_shoez_modules(source: str) -> set[str]:
    """Every shoez module the fixture imports, however it is spelled."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(
                alias.name
                for alias in node.names
                if alias.name == "shoez" or alias.name.startswith("shoez.")
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module == "shoez" or (module or "").startswith("shoez."):
                modules.add(str(module))
    return modules


def test_imports_only_public_names() -> None:
    """The fixture reaches into no private shoez module.

    This is the check that makes "the public API is sufficient" a fact rather
    than a claim. Note it is scoped to the fixture alone: the rest of the test
    suite imports privates freely and legitimately, because it is testing them.
    """
    imported = _imported_shoez_modules(FIXTURE_SOURCE.read_text())
    assert imported, "no shoez imports found; has the fixture moved?"
    forbidden = imported - _ALLOWED_MODULES
    assert not forbidden, (
        f"the out-of-tree step fixture imports private shoez modules: "
        f"{sorted(forbidden)}. Every name a plugin needs must be reachable from "
        f"the top-level `shoez` namespace; export it rather than relaxing this."
    )


def test_every_imported_name_is_public() -> None:
    """Each name pulled from ``shoez`` is in ``__all__``, not merely reachable.

    A name can be importable by accident of module layout while not being part
    of the promised API. ``__all__`` is the promise, so that is what we check.
    """
    import shoez

    tree = ast.parse(FIXTURE_SOURCE.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "shoez"
        for alias in node.names
    }
    assert names, "the fixture imports no names from shoez"
    missing = names - set(shoez.__all__)
    assert not missing, (
        f"the fixture imports {sorted(missing)} from shoez, but they are not in "
        f"shoez.__all__; add them to the public API."
    )


class TestDemoMaxAbsConformance(StepConformanceTest):
    """The out-of-tree step runs the same contract shoez's own steps do."""

    training_data = {
        "a": [1.0, -2.0, 4.0],
        "b": [10.0, 20.0, -40.0],
        "g": ["x", "y", "x"],
    }

    def make_step(self) -> StepDemoMaxAbs:
        return StepDemoMaxAbs(["a", "b"])


def test_works_through_a_recipe() -> None:
    """The end a user actually touches: specify, fit, transform."""
    df = pd.DataFrame({"a": [1.0, -2.0, 4.0], "b": [10.0, 20.0, -40.0]})

    recipe = Recipe().add(StepDemoMaxAbs(s.numeric_features()))
    fitted = recipe.fit(df)
    result = fitted.transform(df)

    assert result["a"].tolist() == pytest.approx([0.25, -0.5, 1.0])
    assert result["b"].tolist() == pytest.approx([0.25, 0.5, -1.0])
