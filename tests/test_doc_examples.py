"""Execute every ``{python}`` docstring example cell and fail if one errors.

great-docs (via Quarto) *runs* ```` ```{python} ```` example cells at build
time and embeds their real output, so a broken example is a build failure
rather than stale rendered text (see issue #118). That validation only happens
during ``great-docs build``, which is a heavy job (installs Quarto + renders
the whole site). This test reproduces just the *execution* half cheaply and
without Quarto, so a broken example fails fast in the normal PR test matrix
instead of only surfacing in the docs build.

The behavior mirrors Quarto's: each reference page runs in its own kernel, so
all cells of a single docstring share one namespace and run in source order (a
hidden ``#| echo: false`` setup cell followed by the visible call). We collect
docstrings straight from the source with ``ast`` — that gives exact source
ordering and a line number for failures, and sidesteps import/inheritance
quirks of runtime introspection.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "shoez"

# ```{python}\n ... \n``` — the fenced executable cell great-docs runs. The
# ``{python}`` marker (with braces) is what distinguishes an executable cell
# from a plain ```` ```python ```` display block, which is rendered verbatim.
_CELL_RE = re.compile(r"^```\{python\}[^\n]*\n(.*?)\n```", re.DOTALL | re.MULTILINE)


def _iter_docstrings(path: Path) -> list[tuple[str, int, str]]:
    """Yield ``(qualname, lineno, docstring)`` for every docstring in ``path``.

    Walks the module, its classes, and its functions/methods (nested included),
    building a dotted qualname so a failure points at the exact object.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    results: list[tuple[str, int, str]] = []

    module_doc = ast.get_docstring(tree, clean=False)
    if module_doc is not None:
        results.append((path.stem, 1, module_doc))

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = f"{prefix}.{child.name}"
                doc = ast.get_docstring(child, clean=False)
                if doc is not None:
                    results.append((qualname, child.lineno, doc))
                visit(child, qualname)

    visit(tree, path.stem)
    return results


def _collect_cells() -> list[tuple[str, str]]:
    """Find every docstring holding at least one executable cell.

    Returns ``(id, source)`` pairs where ``source`` is the docstring's cells
    concatenated in order into one script — the unit Quarto would execute for
    that reference page.
    """
    cases: list[tuple[str, str]] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        for qualname, lineno, doc in _iter_docstrings(path):
            # Docstring cells are indented to the docstring's column; cleandoc
            # strips that common leading whitespace so the fence sits at column
            # 0 where the (line-anchored) cell regex can find it.
            cells = _CELL_RE.findall(inspect.cleandoc(doc))
            if cells:
                cases.append((f"{qualname}:{lineno}", "\n".join(cells)))
    return cases


_CASES = _collect_cells()


def test_docstring_examples_were_found() -> None:
    """Guard the collector itself: the package has executable examples.

    If this drops to zero the discovery has silently broken (e.g. the fence
    format changed) and every per-example check below would vacuously pass.
    """
    assert _CASES, "no ```{python}``` docstring cells were discovered under shoez/"


@pytest.mark.parametrize("source", [c[1] for c in _CASES], ids=[c[0] for c in _CASES])
def test_docstring_example_executes(source: str) -> None:
    """Each docstring's ``{python}`` cells run without raising.

    Cells share one namespace and run in order, exactly as great-docs executes
    the reference page. We only assert the code does not raise; rendering the
    output is the docs build's job.
    """
    namespace: dict[str, object] = {}
    exec(compile(source, "<docstring-example>", "exec"), namespace)
