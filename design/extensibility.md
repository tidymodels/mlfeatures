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
[project.entry-points."pyrecipes.steps"]
step_woe = "mypackage.steps:WoeStep"
step_embed = "mypackage.steps:EmbedStep"
```

The core never relies on import order for correctness. Each plugin step declares its `required_pkgs()` and lazy-imports them.

---

## Base classes

```python
from pyrecipes import Step, Check, RowOp
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
| `pyrecipes-embed`     | Supervised/target encoders, WoE, entity embeddings |
| `pyrecipes-text`      | Tokenization, hashing, NLP steps                   |
| `pyrecipes-imbalance` | Upsampling, downsampling, class-balance steps      |

---

## Step template

A cookiecutter / template is provided covering: `fit`, `transform`, `schema_in`, `schema_out`, `tidy`, `required_pkgs`, `StepConformanceTest` invocation, and a minimal test fixture.

---

## `required_pkgs()`

Each step declares optional heavy dependencies. The core lazy-imports them only when the step is actually used:

```python
def required_pkgs(self) -> list[str]:
    return ["scikit-learn>=1.3"]
```

A clear error is raised at step construction if a declared package is missing.
