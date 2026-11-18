# Serialization

→ See also: [ecosystem.md](ecosystem.md), [extensibility.md](extensibility.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Requirements

A trained recipe must serialize so it can be:
- Saved alongside a model.
- Loaded in a different process for inference.
- Diffed and reviewed in code review (stable JSON/YAML format covering hyperparameters and learned stats).

---

## No closures, no environments

Steps store **schemas and fitted statistics**, never raw training data. No closures captured into pickled state.

R `recipes` captured entire `globalenv()` references via formula quosures, inflating saved recipes and requiring the `butcher` package to clean them up afterward [#302, #400, #503, #760, #859]. We avoid this by design:

- Steps store statistics as plain Python objects (scalars, arrays, DataFrames).
- Selectors are stored as serializable expression objects, not closures.
- Environment-dependent settings (locale, contrast schemes) are captured at `fit()` time as plain values on the step.

---

## Pickle strategy

Step state is a **frozen dataclass of plain fields** — no methods serialized. The class definition is looked up by qualified name at load time. Missing fields (from an older step version) take class defaults on load.

```python
@dataclass(frozen=True)
class NormalizeStep(Step):
    means: dict[str, float]
    stds: dict[str, float]
    # ... selector and hyperparams
```

This gives us the best of both worlds:
- New arguments added in code default cleanly on load (no `if is None` shims).
- Major versions are still allowed to break and require re-fit — and this is documented loudly.

---

## `recipe.minimize()`

Strips training artifacts that are not needed for `transform()` — for example, the KNN imputer's retained neighbor matrix. Available from day one. Steps that need training data at predict time (e.g. `step_impute_knn`) explicitly retain only the columns they need, not the whole data frame.

---

## Back-compat policy

Decided explicitly before v0.1 (R `recipes` never did this and paid for it [#956, #1093, #1167, #1349]):

- **Within a major version:** missing fields on load take class defaults. No shims required.
- **Across major versions:** re-fit is required. Documented loudly in release notes.
- **No cross-major pickle compatibility guarantee.**

See [goals.md § open questions](goals.md) for the deprecation policy decision.

---

## Loading a payload never runs code the payload picks

A JSON or YAML payload is data from outside the process, so the class names in it are requests, not instructions. Three places name a class: a step's `"type"`, a fit-state `"__dataclass__"`, and a selector's `"step_type"`. While `from_dict` decodes a payload (and so `from_json` and `from_yaml`, which share the decoder), all three resolve against a fixed trust set:

- `mlfeatures` and its submodules.
- Packages that register a step under the `mlfeatures.steps` entry-point group. That is an install-time declaration made by the plugin's own distribution, not something the payload can assert.
- Packages of the classes the caller passes as `trusted` (see [Custom step serialization](#custom-step-serialization)).

The trust unit is a top-level package, and it is checked twice. The name is checked **before** its module is imported, because importing runs code too. The resolved class is then checked against the module that *defines* it, so a trusted module that re-exports a third-party class does not make that class loadable. A step re-exported from a package other than the one that defines it is found through its entry point instead.

The resolved object must also be the right kind of thing: a `Step` subclass for `"type"` and `"step_type"`, a dataclass for `"__dataclass__"`. A name outside the trust set (`builtins.print`, `subprocess.call`) raises `ValueError`, as does a malformed payload (a non-string name, `"fields"` that is not a mapping). Using `yaml.safe_load` alone would not have been enough, because the decoder is shared.

State keys are checked too: `Step.from_state` refuses a key that would replace a method, property, or dunder attribute of the step. Anything else is restored as given, since the payload *is* the step's state.

The allow-list applies to payloads, not to state produced in this process. `Step.from_state` called directly on a step's own `Step.to_state` (the `minimize()` idiom) is unrestricted, because nothing outside the process chose the names. It is not an entry point for untrusted data: load that through `from_dict`.

This guards against a payload choosing what to call. It does not validate the values inside a payload, and a registered plugin step still runs its own author's code at transform time. Pickle is outside this guarantee: unpickling executes arbitrary code by design, so only unpickle recipes you trust.

---

## Custom step serialization

Custom user steps (plugin packages) register via the entry-point mechanism so deserialization can locate them by qualified name. See [extensibility.md](extensibility.md). A registered step loads with no further ceremony.

A step that is not registered, such as one defined in a script or notebook or one still under development, is refused on load unless the caller names it:

```python
fitted = FittedRecipe.from_json(text, trusted=[MyStep])
```

`trusted` takes classes, and trusts their top-level packages for that one call, so a fit dataclass defined next to the step loads too. Pass only classes you wrote or trust: a package listed there is imported as the payload asks. `trusted` is accepted by `from_dict`, `from_json`, and `from_yaml`, on the module functions and on `FittedRecipe`. Pickle is the alternative only for a file you trust, and it carries the risk above.

Saving warns when a recipe names a class that `from_dict` refuses by default (an unregistered step, or a fit dataclass from an unregistered package). The payload is still written, since `trusted` can load it, but the mismatch shows up when the file is saved rather than the first time it is loaded, possibly in another process.

`StepConformanceTest` runs two JSON checks. `test_json_preserves_transform_output` round-trips the step with itself passed as `trusted`, so a step under development is exercised before it is registered. `test_json_loads_without_trusting_the_step` loads it with no `trusted` argument, which is what a user of the step sees. It passes for steps in `mlfeatures` and for a plugin step once its entry point is declared and installed. Until then it skips and names the reason, so a missing or mistyped entry point is visible in `pytest -rs` rather than discovered by the first user.
