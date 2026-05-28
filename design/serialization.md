# Serialization

→ See also: [ecosystem.md](ecosystem.md), [extensibility.md](extensibility.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Requirements

A prepped recipe must serialize so it can be:
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

Strips training artifacts that are not needed for `transform()` — for example, the KNN imputer's retained neighbor matrix. Available from day one. Steps that need training data at predict time (e.g. `step_impute_knn`) explicitly retain only the columns they need, not the whole frame.

---

## Back-compat policy

Decided explicitly before v0.1 (R `recipes` never did this and paid for it [#956, #1093, #1167, #1349]):

- **Within a major version:** missing fields on load take class defaults. No shims required.
- **Across major versions:** re-fit is required. Documented loudly in release notes.
- **No cross-major pickle compatibility guarantee.**

See [goals.md § open questions](goals.md) for the deprecation policy decision.

---

## Custom step serialization

Custom user steps (plugin packages) must register via the entry-point mechanism so deserialization can locate them by qualified name. See [extensibility.md](extensibility.md).
