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
[project.entry-points."mlfeatures.steps"]
step_woe = "mypackage.steps:WoeStep"
step_embed = "mypackage.steps:EmbedStep"
```

The core never relies on import order for correctness. Each plugin step declares its `required_pkgs()` and lazy-imports them.

Registration is also what lets `from_json`, `from_yaml`, and `from_dict` load a recipe containing the step: they accept only classes from `mlfeatures`, from registered packages, and from classes the caller passes as `trusted`. An unregistered step is refused on load. See [serialization.md](serialization.md#loading-a-payload-never-runs-code-the-payload-picks).

---

## Base classes

```python
from mlfeatures import Step, RowOp
```

- `Step` — base for all transformation steps. See [step-contract.md](step-contract.md) for the full ABC.
- `RowOp(Step)` — base for row-altering steps; skipped at transform time automatically.

**Compose-don't-subclass** is the recommended extension pattern, but implementing the `Step` ABC directly is fully supported and documented. Subclassing `Step` subclasses is explicitly not supported (R explicitly refused it [#1058]); extend by implementing the ABC.

---

## Initializing a step

A step's `__init__` stores its own arguments and delegates the framework-owned ones to `Step.__init__()`, which sets `id`, `trained`, `output_role`, and the value behind the read-only `keep_original_cols`. The class also declares its `mode`, which the base class reads; a step that omits it cannot be constructed.

```python
from mlfeatures import Mode, Selector, Step

class StepMyThing(Step):
    mode = Mode.ONE_TO_ONE

    def __init__(self, columns: Selector, *, id: str | None = None) -> None:
        self.columns = columns
        super().__init__(id, "my_thing")
```

The second argument is the id to default to when the caller passes none; `keep_original_cols` and `output_role` are keyword-only and both default to `None`. A plugin never assigns those attributes itself, and needs no private name to set them.

`keep_original_cols` is worth passing only from a step whose `mode` is `ONE_TO_MANY` or `MANY_TO_MANY`; for any other mode the base class derives the value from the mode and the attribute is read-only, so a step neither states it nor can change it later. See [Constructors](step-contract.md#constructors) for the full rules.

---

## Resolving a step's own selection

A step's `columns` argument holds a selector, not column names, so `fit()` has to resolve it. `Step.resolve_selection()` is the supported route; narwhals' own `frame.schema` supplies the dtypes:

```python
from mlfeatures import Frame, SchemaInfo, Step

class StepMyThing(Step):
    def fit(self, training_data: Frame, info: SchemaInfo) -> "StepMyThing":
        names = self.resolve_selection(training_data, info)
        selected = training_data.select(names)
        dtypes = {name: str(dt) for name, dt in selected.schema.items()}
        ...
```

mlfeatures's own steps reach for the private `mlfeatures._frame` here instead, for historical reasons. **A plugin does not need it**, and should not import it: `resolve_selection()` covers the selector half and `frame.schema` covers the dtype half. The path above is the one that type-checks under `mypy --strict` and `pyright --strict` using only public names.

Annotating the ABC's methods needs two type aliases that are exported alongside the base classes: `Selector` (the type of every step's `columns` parameter) and `AcceptedDtypes` (the return type of `accepts_dtypes()`).

---

## Plugin tier namespaces

Namespace-reserved plugin packages — see [steps-catalog.md](steps-catalog.md):

| Package               | Covers                                             |
| --------------------- | -------------------------------------------------- |
| `mlfeatures-embed`     | Supervised/target encoders, WoE, entity embeddings |
| `mlfeatures-text`      | Tokenization, hashing, NLP steps                   |
| `mlfeatures-imbalance` | Upsampling, downsampling, class-balance steps      |

---

## Conformance

`mlfeatures.testing.StepConformanceTest` ships inside the package, so a plugin runs the same contract mlfeatures's own steps do:

```python
from mlfeatures.testing import StepConformanceTest

class TestMyStepConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepMyThing:
        return StepMyThing(["a"])
```

It needs pytest, which mlfeatures does not require at runtime: install the `test` extra (`pip install mlfeatures[test]`). See [step-contract.md](step-contract.md) for the checks and the opt-out hooks.

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
`mlfeatures_woe` is required by step 'woe_1' (step_woe) but is not installed; install with `pip install mlfeatures_woe`.
```

Version constraints are reported in the hint but not enforced: mlfeatures does not vendor a version parser, and a wrong-version library fails at the call site anyway. See [dependencies.md](dependencies.md) for the lazy-import tools a step's body uses to reach those packages.
