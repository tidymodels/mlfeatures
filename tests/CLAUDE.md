# tests

## Per-step test files

Each step function has its own series of test files. For a step `step_x`, there are three:

- `test_step_x.py` — unit tests for the step's own logic (fit/transform behavior, estimates, guards, edge cases).
- `test_step_x_e2e.py` — end-to-end tests exercising the step through a recipe.
- `test_step_x_infrastructure.py` — conformance/infrastructure tests (the `StepConformanceTest` contract: pickle round-trip, stable `id`, `produces_sparse`, dtype acceptance, schema_in/out, etc.).

## Infrastructure test coverage

The conformance contract lives in `shoez.testing.StepConformanceTest` and is written **once**. A `test_step_x_infrastructure.py` file supplies only the step and its data:

```python
from shoez import StepNormalize
from shoez.testing import StepConformanceTest


class TestNormalizeConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepNormalize:
        return StepNormalize(["a"])
```

Uniformity is now structural — every step inherits the same checks, so there is nothing to keep in sync by hand and no `grep` to run. Adding a check to the contract means editing `shoez/testing.py` once, and every step picks it up.

If a check genuinely does not apply to a step, opt out through a hook and say why in a comment — never by omitting anything, since there is nothing to omit:

- `make_empty_step()` returning `None` — the step takes no column selector (`step_intercept`, `step_filter`, `step_mutate`).
- `estimates_attr = None` — the step stores no fit-time estimates.
- `supports_json = False` — the step refuses JSON serialization (`step_mutate`); the check then asserts the refusal.

Each of these skips with an explicit reason, visible in `pytest -rs`, so "considered and not applicable" stays distinguishable from "forgotten".

Because `shoez.testing` ships inside the package, third-party steps run this same contract; see `design/extensibility.md`.

## The extensibility fixture

`tests/_plugin_fixture.py` defines `StepScaleShift`, a stand-in for a third-party step, and `tests/test_extensibility.py` pushes it through every surface a plugin uses. It exists so that breaking the extensibility path fails CI here, rather than being discovered later by someone building against the package.

The fixture is **not** a step shoez ships, and it must stay writable against `shoez.__all__` alone. `TestPublicApiOnly` parses the file and fails on any import from a private module that is not in `PRIVATE_IMPORT_ALLOWLIST`. That allowlist is a ratchet in both directions:

- adding an entry means a plugin author now needs a private import too, so it must cite the issue tracking the gap;
- removing the need for one is caught as well — `test_allowlist_is_still_needed` fails once an allowlisted name becomes public, so the list cannot rot.

When changing the fixture, keep it honest: it should do what a plugin can do, not what shoez can do. If a change makes the fixture need something new from shoez, that is the finding, not an inconvenience to work around.

The fixture shares an interpreter and source tree with shoez, so it cannot cover problems that only appear across a distribution boundary — packaging, editable installs, type-checker resolution, real entry-point discovery. Those still need an out-of-tree package.

## Input dtype coverage

Steps that accept multiple input dtypes — like `step_normalize`, which accepts both floats and integers — must have unit tests verifying they work with each accepted dtype.

These same tests must also assert that the output dtype is always float, even when an integer output could happen to be produced by chance.
