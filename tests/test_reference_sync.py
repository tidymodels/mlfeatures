"""Guard that the docs reference cannot drift from ``mlfeatures.__all__``.

``CLAUDE.md`` requires the ``reference:`` section in ``great-docs.yml`` to
mirror the public API. great-docs ``contents:`` does not support regex/glob
(items are literal names passed to quartodoc), so the sections are maintained
by hand and can silently drift. This test fails the instant a public export is
added without giving it a reference page.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import yaml

import mlfeatures

GREAT_DOCS_YML = Path(__file__).resolve().parent.parent / "great-docs.yml"

# ``__version__`` is a module attribute, not an API object with a page.
_NOT_DOCUMENTED = {"__version__"}

# Submodules documented on their own reference page. They are reached as
# ``mlfeatures.<name>`` and are deliberately not in ``mlfeatures.__all__`` (which is the
# step-authoring API), so the reverse-direction check must allow them.
_SUBMODULES = {"data"}


def _documented_names() -> set[str]:
    """Collect every top-level name referenced in ``great-docs.yml``.

    Each ``contents`` entry is either a bare string (``Recipe.add``) or a
    mapping with a ``name:`` key (``{name: Recipe, members: false}``). We take
    the leading dotted segment so method entries (``Recipe.add``) count toward
    documenting their owning class (``Recipe``).
    """
    return {name.split(".", 1)[0] for name in _all_entries()}


def _all_entries() -> set[str]:
    """Every ``contents`` entry verbatim, dotted method names included."""
    config: dict[str, Any] = yaml.safe_load(GREAT_DOCS_YML.read_text())
    names: set[str] = set()
    for section in config["reference"]:
        for item in section["contents"]:
            raw = item["name"] if isinstance(item, dict) else item
            names.add(str(raw))
    return names


def test_every_public_export_has_a_reference_page() -> None:
    expected = set(mlfeatures.__all__) - _NOT_DOCUMENTED
    documented = _documented_names()
    missing = expected - documented
    assert not missing, (
        f"public exports missing from great-docs.yml reference: {sorted(missing)}. "
        "Add them to the appropriate `reference:` section."
    )


def test_no_reference_page_for_a_deleted_export() -> None:
    """The reverse direction: nothing is documented that is no longer exported.

    Without this, renaming or removing a public name leaves a dangling
    ``reference:`` entry that only surfaces as a docs-build failure on ``main``.
    """
    documented = _documented_names()
    extra = documented - set(mlfeatures.__all__) - _SUBMODULES
    assert not extra, (
        f"great-docs.yml documents names that are not in mlfeatures.__all__: "
        f"{sorted(extra)}. Remove the entry, or re-export the name."
    )


def test_step_authoring_methods_are_all_documented() -> None:
    """Every public method `Step` itself defines has a reference row.

    `Step` is the extension contract — a plugin author implements or calls each
    of these — so its own surface is enumerated by hand in the ``Developing``
    section and must stay complete. Issue #197 found ``resolve_selection``
    missing precisely because the check above truncates ``Step.fit`` to
    ``Step`` and so proves nothing about members.

    Scoped to `Step` deliberately. The concrete steps inherit this entire
    surface and are documented with ``members: false`` (their class docstring is
    the page), so applying the same rule to them would demand ~15 redundant rows
    each.
    """
    own_methods = {
        name
        for name, value in vars(mlfeatures.Step).items()
        if not name.startswith("_")
        and (
            inspect.isfunction(value)
            or isinstance(value, (property, classmethod, staticmethod))
        )
    }
    documented = _all_entries()
    missing = {m for m in own_methods if f"Step.{m}" not in documented}
    assert not missing, (
        f"public Step methods missing from the `Developing` section of "
        f"great-docs.yml: {sorted(missing)}. A plugin author needs each one; "
        f"add a `Step.<name>` row."
    )


def test_data_submodule_has_a_reference_page() -> None:
    """``mlfeatures.data`` is not in ``__all__``, so the forward check skips it."""
    assert "data" in _documented_names(), (
        "add a `Datasets` section to great-docs.yml listing `data`"
    )
