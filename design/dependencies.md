# Optional dependencies

shoez keeps a **small required core** and treats heavier libraries as
*optional*. Only `narwhals` (the internal dataframe representation — see
[ecosystem.md](ecosystem.md)) is unconditionally required. `numpy`, `pandas`,
`polars`, and `scikit-learn` are optional: a user who only needs a subset of the
steps should not be forced to install all of them, and `import shoez` should
never fail merely because one of them is absent.

The single place this is coordinated is [`shoez/_dependencies.py`](../shoez/_dependencies.py).
This document is the guide for **step authors**: how to depend on an optional
library from inside a step without breaking `import shoez` for users who do not
have it installed.

> **Status (transitional).** The machinery described here is in place and
> tested. Existing steps still `import numpy` at module top level, and
> `pyproject.toml` still lists these libraries as required dependencies. Both
> are being migrated incrementally — adopt the patterns below in new or
> touched steps; we are intentionally *not* rewriting every step in one pass, to
> avoid merge conflicts with in-flight step work.

## The two tools

`shoez._dependencies` exposes two ways to reach an optional library, for two
different situations.

### 1. Lazy module proxy — for ordinary `np.<...>` call sites

Import the proxy instead of the real module:

```python
from __future__ import annotations

from shoez._dependencies import numpy as np
```

The proxy is transparent: every `np.asarray(...)`, `np.mean(...)`, etc. works
exactly as before. The real `numpy` import is deferred until the **first
attribute access** — i.e. until a step actually runs — so importing the step
module (and therefore `import shoez`) stays cheap and succeeds even when numpy
is not installed. If numpy *is* missing, the first `np.<...>` raises a
`MissingDependencyError` with an install hint:

```
`numpy` is required but is not installed; install it with `pip install 'shoez[numpy]'` (or `pip install numpy`).
```

Use this for the common case where a step calls top-level functions on the
module.

### 2. `import_optional(...)` — for everything the proxy can't cover

`import_optional` eagerly imports a module (or raises the same friendly error).
Reach for it when a lazy attribute proxy doesn't fit:

**A specific submodule or symbol.** The proxy only forwards attribute access on
the top-level module; importing a deep submodule or a named symbol is clearer
done explicitly, inside the function that needs it:

```python
def apply_polynomial(features: Frame, ...) -> Frame:
    polynomial = import_optional(
        "numpy.polynomial.polynomial", purpose="to build the polynomial basis"
    )
    vander = polynomial.polyvander(...)
    ...
```

Keeping the import inside the function body (not at module top level) is what
defers it. The optional `purpose=` fragment is woven into the error message.

**Guarding a class definition.** A class that *subclasses* an optional
library's types (e.g. the scikit-learn adapter subclasses `BaseEstimator`)
cannot use the lazy proxy, because the base classes must resolve when the class
body executes. Guard the import and fall back to placeholder bases, then fail
loudly only on *use*:

```python
try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _HAS_SKLEARN = True
except ModuleNotFoundError:
    _HAS_SKLEARN = False

    class BaseEstimator:  # placeholder so the class body below can run
        ...

    class TransformerMixin:
        ...


class RecipeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, recipe: Recipe):
        if not _HAS_SKLEARN:
            # Re-runs the failed import so the error carries the install hint.
            import_optional("sklearn", purpose="to use RecipeTransformer")
        self.recipe = recipe
```

`MissingDependencyError` subclasses `ModuleNotFoundError`, so any existing
`except ModuleNotFoundError` / `except ImportError` continues to catch it.

**A sanctioned runtime backend import.** The core is backend-agnostic and must
not `import polars` / `import pandas` at runtime to branch on backend (see
`design/ecosystem.md`, boundary rule #5). The few *deliberate* exceptions —
e.g. the opt-in `shoez.polars` accessor, and resolving a polars selector in
`shoez._frame` — route through `import_optional` rather than a bare `import`:

```python
else:  # runtime; TYPE_CHECKING branch does `import polars as pl` for the checkers
    pl = import_optional("polars", purpose="for the `df.shz` accessor")
```

This is as much for **auditability** as for the friendly error: it makes every
real runtime-backend import greppable as `import_optional("polars"...)` /
`import_optional("pandas"...)`, so the boundary stays reviewable. (In some spots
— like `_frame`'s selector expansion — the missing-module branch is actually
unreachable because the surrounding type check only matches when the backend is
already loaded; the call is kept uniform anyway, with a comment saying so.)

## Type annotations

`from __future__ import annotations` makes all annotations strings, so they are
never evaluated at runtime. That means type-only imports must **not** run at
import time — put them under `TYPE_CHECKING`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


def fit_thing(...) -> "npt.NDArray[Any]": ...
```

`shoez._dependencies` itself imports the real `numpy` under `TYPE_CHECKING`, so
`from shoez._dependencies import numpy as np` gives mypy and pyright (both run in
strict mode) the true module types while the runtime gets the lazy proxy.

## Adding a new optional dependency

1. Add the top-level module name to `_EXTRAS` in
   [`shoez/_dependencies.py`](../shoez/_dependencies.py), mapping it to the pip
   *extra* that will provide it. This makes the install hint correct.
2. If it is a "call lots of top-level functions" library like numpy, add a lazy
   proxy next to `numpy` (`mylib = _lazy_import("mylib")`) and list it in
   `__all__`. Otherwise just use `import_optional(...)` at the call site.
3. When the dependency is migrated out of the required set, add the matching
   `[project.optional-dependencies]` extra in `pyproject.toml` (and the dev/test
   setup so CI installs everything).

## Testing

The mechanism is covered by [`tests/test_dependencies.py`](../tests/test_dependencies.py).
When a step adopts an optional dependency, prefer asserting on the
*behavior*: that the relevant call raises `MissingDependencyError` when the
library is unavailable (you can construct a proxy for a deliberately-absent
module name to exercise the missing path without uninstalling anything), and
that the install hint names the right extra.
