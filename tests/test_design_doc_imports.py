"""Guard that ``mlfeatures`` imports written in ``design/`` actually resolve.

``design/`` uses plain ```` ```python ```` display fences, not the executable
```` ```{python} ```` cells that ``test_doc_examples.py`` runs, so nothing has
ever checked them. That is how ``design/extensibility.md`` came to open its
base-classes section with ``from mlfeatures import Step, Check, RowOp`` when no
``Check`` has ever existed (see issue #197): a plugin author following the doc
literally wrote an import that fails.

We deliberately do **not** execute these blocks. Most are fragments or elide
setup, so running them would fail for reasons that say nothing about the docs'
accuracy. Import resolution is the checkable subset: it is exactly the claim
"this name is part of mlfeatures", which is the claim that went stale.

Private modules are allowed here. ``design/`` documents internals, so
``from mlfeatures._dependencies import numpy`` is legitimate; the check is that the
name *exists*, not that it is public.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

DESIGN_DIR = Path(__file__).resolve().parent.parent / "design"

# ```python\n ... \n```: the plain display fence used throughout design/.
_BLOCK_RE = re.compile(r"^```python[^\n]*\n(.*?)\n```", re.DOTALL | re.MULTILINE)

# `interfaces.md` weighs *candidate* construction interfaces against each other;
# its blocks spell out proposed APIs (`from mlfeatures.steps import ImputeMedian`)
# that were considered and not adopted. Those imports are supposed not to
# resolve: rewriting them to the shipped names would destroy the comparison the
# document exists to make.
_DELIBERATIVE = {"interfaces.md"}


def _mlfeatures_imports(source: str) -> list[tuple[str, str | None]]:
    """Extract ``(module, attribute)`` pairs for every mlfeatures import in a block.

    ``import mlfeatures.foo`` yields ``("mlfeatures.foo", None)``; ``from mlfeatures import X``
    yields ``("mlfeatures", "X")``. Blocks that do not parse are skipped by the
    caller, so this only ever sees syntactically valid Python.
    """
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "mlfeatures" or alias.name.startswith("mlfeatures."):
                    found.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            if module is None or not (
                module == "mlfeatures" or module.startswith("mlfeatures.")
            ):
                continue
            for alias in node.names:
                found.append((module, alias.name))
    return found


def _collect() -> list[tuple[str, str, str | None]]:
    """Yield ``(case_id, module, attribute)`` for every mlfeatures import in design/."""
    cases: list[tuple[str, str, str | None]] = []
    for path in sorted(DESIGN_DIR.glob("*.md")):
        if path.name in _DELIBERATIVE:
            continue
        for index, block in enumerate(_BLOCK_RE.findall(path.read_text())):
            try:
                imports = _mlfeatures_imports(block)
            except SyntaxError:
                # A fragment or pseudo-code block; nothing to check.
                continue
            for module, attribute in imports:
                target = module if attribute is None else f"{module}.{attribute}"
                cases.append((f"{path.name}[{index}]:{target}", module, attribute))
    return cases


_CASES = _collect()


def test_design_doc_imports_were_found() -> None:
    """Guard the collector: if discovery breaks, every check below passes vacuously."""
    assert _CASES, "no mlfeatures imports were discovered in design/*.md"


@pytest.mark.parametrize(
    ("module", "attribute"),
    [(c[1], c[2]) for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_design_doc_import_resolves(module: str, attribute: str | None) -> None:
    """Each name a design doc imports from mlfeatures exists."""
    try:
        imported = importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - only on a stale doc
        pytest.fail(f"design/ imports {module!r}, which does not exist: {exc}")
    if attribute is not None:
        assert hasattr(imported, attribute), (
            f"design/ imports {attribute!r} from {module!r}, but it does not exist"
        )
