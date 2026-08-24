# Extensibility

→ See also: [step-contract.md](step-contract.md), [serialization.md](serialization.md), [steps-catalog.md](steps-catalog.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Third-party packages must be able to define new steps without monkey-patching.

---

## Plugin registration via entry points

Extensions register steps using Python's standard `importlib.metadata` entry points — order-independent and far more reliable than R's S3 dispatch, which broke under package load-order issues [#592]:

```toml
# In a plugin package's pyproject.toml:
[project.entry-points."shoez.steps"]
step_woe = "mypackage.steps:WoeStep"
step_embed = "mypackage.steps:EmbedStep"
```

The core never relies on import order for correctness. Each plugin step declares its `required_pkgs()` and lazy-imports them.

---

## Base classes

```python
from shoez import Step, Check, RowOp
```

- `Step` — base for all transformation steps. See [step-contract.md](step-contract.md) for the full ABC.
- `Check` — base for validation steps (pure functions; must not mutate state).
- `RowOp(Step)` — base for row-altering steps; skipped at transform time automatically.

**Compose-don't-subclass** is the recommended extension pattern, but implementing the `Step` ABC directly is fully supported and documented. Subclassing `Step` subclasses is explicitly not supported (R explicitly refused it [#1058]); extend by implementing the ABC.

---

## Plugin tier namespaces

Namespace-reserved plugin packages — see [steps-catalog.md](steps-catalog.md):

| Package               | Covers                                             |
| --------------------- | -------------------------------------------------- |
| `shoez-embed`     | Supervised/target encoders, WoE, entity embeddings |
| `shoez-text`      | Tokenization, hashing, NLP steps                   |
| `shoez-imbalance` | Upsampling, downsampling, class-balance steps      |

---

## Conformance

`shoez.testing.StepConformanceTest` ships inside the package, so a plugin runs the same contract shoez's own steps do:

```python
from shoez.testing import StepConformanceTest

class TestMyStepConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepMyThing:
        return StepMyThing(["a"])
```

It needs pytest, which shoez does not require at runtime: install the `test` extra (`pip install shoez[test]`). See [step-contract.md](step-contract.md) for the checks and the opt-out hooks.

---

## Step template

A cookiecutter / template is **not yet provided**. It would cover: `fit`, `transform`, `schema_in`, `schema_out`, `required_pkgs`, a `StepConformanceTest` subclass, and a minimal test fixture.

---

## `required_pkgs()`

Each step declares optional heavy dependencies. The core lazy-imports them only when the step is actually used:

```python
def required_pkgs(self) -> list[str]:
    return ["scikit-learn>=1.3"]
```

Each entry is a requirement string: a bare package name, or one carrying a version constraint. A name that differs from its importable module (`scikit-learn` → `sklearn`) is mapped; otherwise dashes become underscores.

`Step.check_required_pkgs()` turns the declaration into a check. The recipe calls it at the two points where a step actually runs: `Recipe.fit()` and `FittedRecipe.transform()`. Building a recipe, inspecting it, and serializing it never need the package, so they are not gated; and checking at both ends means a recipe deserialized somewhere the original environment's packages are absent still fails with the actionable error rather than an `ImportError` from inside a step. A missing package raises `MissingDependencyError` naming the step, the package, and the `pip install` that fixes it:

```
`shoez_woe` is required by step 'woe_1' (step_woe) but is not installed; install with `pip install shoez_woe`.
```

Version constraints are reported in the hint but not enforced: shoez does not vendor a version parser, and a wrong-version library fails at the call site anyway. See [dependencies.md](dependencies.md) for the lazy-import tools a step's body uses to reach those packages.
