# this module is adapted from polars: https://github.com/pola-rs/polars/blob/main/py-polars/src/polars/_dependencies.py
# MIT license below.
# Copyright (c) 2025 Ritchie Vink
# Copyright (c) 2024 (Some portions) NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Optional-dependency machinery.

shoez keeps a small required core (narwhals) and treats heavier libraries —
numpy, pandas, polars, scikit-learn — as *optional*. This module is the single
place that turns "the library might be absent" into a predictable, friendly
failure instead of an ``ImportError`` at ``import shoez`` time.

Two tools, for two situations:

- `import_optional()` — eager import-or-raise. Use it where you need the
  module object right now (e.g. inside a step's ``fit``/``apply`` body, or when
  guarding a class definition). On a missing module it raises
  `MissingDependencyError` with an install hint.

- `numpy` (and friends, via `_lazy_import()`) — a *lazy* module proxy.
  Importing it never fails; the real import is deferred to first attribute
  access, and a missing library surfaces there as `MissingDependencyError`.
  This lets a step write ``from shoez._dependencies import numpy as np`` and keep
  every ``np.<...>`` call site unchanged, while ``import shoez`` stays cheap and
  works even when numpy is not installed.

See ``design/dependencies.md`` for the step-author guide.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # For type-checkers and IDEs, the lazy proxies below are the real modules.
    import numpy

__all__ = [
    "MissingDependencyError",
    "import_optional",
    "module_available",
    "numpy",
]

# Top-level module name -> the shoez extra that provides it, for install hints.
# Listed even when not yet wired up as a lazy proxy so the hint is correct the
# moment a step starts depending on the library.
_EXTRAS: dict[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "polars": "polars",
    "sklearn": "sklearn",
    "scipy": "scipy",
}


class MissingDependencyError(ModuleNotFoundError):
    """An optional dependency was used but is not installed.

    Subclasses `ModuleNotFoundError` so existing ``except
    ModuleNotFoundError`` / ``except ImportError`` guards keep working.
    """


def _install_hint(module_name: str) -> str:
    """A ``pip install`` suggestion for a (possibly dotted) module name."""
    top = module_name.split(".")[0]
    extra = _EXTRAS.get(top)
    if extra is not None:
        return (
            f"install it with `pip install 'shoez[{extra}]'` (or `pip install {top}`)"
        )
    return f"install it with `pip install {top}`"


def _missing_error(
    module_name: str, *, purpose: str | None = None
) -> MissingDependencyError:
    top = module_name.split(".")[0]
    context = f" {purpose}" if purpose else ""
    return MissingDependencyError(
        f"`{top}` is required{context} but is not installed; {_install_hint(module_name)}."
    )


def import_optional(module_name: str, *, purpose: str | None = None) -> ModuleType:
    """Import ``module_name``, or raise `MissingDependencyError`.

    Parameters
    ----------
    module_name
        The import path, e.g. ``"numpy"`` or ``"numpy.polynomial.polynomial"``.
    purpose
        Optional phrase woven into the error message describing why the module
        is needed, e.g. ``"to expand polynomials"``. Keep it a sentence fragment.
    """
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise _missing_error(module_name, purpose=purpose) from exc


def module_available(module_name: str) -> bool:
    """Whether ``module_name`` can be imported, without importing it."""
    if module_name in sys.modules:
        return True
    try:
        spec = importlib.util.find_spec(module_name)
    except ModuleNotFoundError:
        # A parent package along the dotted path is itself missing.
        return False
    # A spec with no loader (e.g. a namespace-package placeholder) is not a
    # module we can actually import and use. (Matches polars' _lazy_import.)
    return spec is not None and spec.loader is not None


class _LazyModule(ModuleType):
    """A stand-in for an optional module whose import is deferred to first use.

    Construction never imports anything. The first access of a public attribute
    triggers the real import via `import_optional()`, so a missing library
    raises `MissingDependencyError` at the call site rather than at
    ``import`` time. Dunder lookups deliberately raise `AttributeError`
    instead, so inspection/tooling probes are not mistaken for usage.
    """

    def __init__(self, module_name: str) -> None:
        super().__init__(module_name)
        # Bypass __setattr__/__getattr__ machinery for our own bookkeeping.
        self.__dict__["_module_name"] = module_name

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        module = import_optional(self.__dict__["_module_name"])
        # Copy the real module's namespace onto the proxy so subsequent attribute
        # access resolves via ordinary __dict__ lookup and never re-enters this
        # method — steps touch ``np.<...>`` in hot loops. (Adapted from polars'
        # _LazyModule._import.)
        self.__dict__.update(module.__dict__)
        return getattr(module, name)


def _lazy_import(module_name: str) -> ModuleType:
    """Return a lazy proxy for ``module_name`` (see `_LazyModule`)."""
    return _LazyModule(module_name)


if not TYPE_CHECKING:
    # Runtime: a lazy proxy. Type-checkers instead see ``import numpy`` above.
    numpy = _lazy_import("numpy")
