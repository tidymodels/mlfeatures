"""Guard that the docs reference cannot drift from ``shoez.__all__``.

``CLAUDE.md`` requires the ``reference:`` section in ``great-docs.yml`` to
mirror the public API. great-docs ``contents:`` does not support regex/glob
(items are literal names passed to quartodoc), so the sections are maintained
by hand and can silently drift. This test fails the instant a public export is
added without giving it a reference page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

import shoez

GREAT_DOCS_YML = Path(__file__).resolve().parent.parent / "great-docs.yml"

# ``__version__`` is a module attribute, not an API object with a page.
_NOT_DOCUMENTED = {"__version__"}


def _documented_names() -> set[str]:
    """Collect every top-level name referenced in ``great-docs.yml``.

    Each ``contents`` entry is either a bare string (``Recipe.add``) or a
    mapping with a ``name:`` key (``{name: Recipe, members: false}``). We take
    the leading dotted segment so method entries (``Recipe.add``) count toward
    documenting their owning class (``Recipe``).
    """
    config: dict[str, Any] = yaml.safe_load(GREAT_DOCS_YML.read_text())
    names: set[str] = set()
    for section in config["reference"]:
        for item in section["contents"]:
            raw = item["name"] if isinstance(item, dict) else item
            names.add(str(raw).split(".", 1)[0])
    return names


def test_every_public_export_has_a_reference_page() -> None:
    expected = set(shoez.__all__) - _NOT_DOCUMENTED
    documented = _documented_names()
    missing = expected - documented
    assert not missing, (
        f"public exports missing from great-docs.yml reference: {sorted(missing)}. "
        "Add them to the appropriate `reference:` section."
    )
