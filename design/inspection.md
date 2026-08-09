# Errors, Printing, and Inspection

→ See also: [step-contract.md](step-contract.md), [schema.md](schema.md), [lifecycle.md](lifecycle.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## `repr(recipe)`

Shows different information depending on training status:

**Before fit** (unresolved):
```
Recipe (untrained)
  Inputs: 12 columns — 1 outcome, 11 predictors
  Steps:
    step_impute_knn(all_predictors())              [untrained]
    step_dummy(all_nominal_predictors())           [untrained]
    step_normalize(all_numeric_predictors())       [untrained]
```

**After fit** (resolved):
```
Recipe (trained)
  Inputs: 12 columns — 1 outcome, 11 predictors
  Steps:
    step_impute_knn [x1, x2, x4, x7]              [trained]
    step_dummy      [color → color_red, color_blue][trained]
    step_normalize  [x1, x2, x3, x5, x6]          [trained]
```

Pre-fit showing unresolved quosures was a known misleading behavior in R `recipes` [#507]. We distinguish the two states explicitly.

---

## Error messages

Step errors are raised via exception chaining so the full traceback is preserved:

```python
raise StepError(
    f"step '{self.id}' ({type(self).__name__}) failed on column 'x1'"
) from underlying_exc
```

Every error includes:
- The step `id`
- The step type
- The offending column(s)

Python's `raise X from e` gives this for free — R needed a deferred fix [#734] that was never fully satisfying.

---

## Zero-match selectors

A selector that matches zero columns is **silent by default** — the step is a no-op and processing continues normally. Steps must handle empty selection without error. The `StepConformanceTest` verifies this path explicitly — see [step-contract.md](step-contract.md).

This is the opposite of R's default, which errors on zero matches and requires `allow_empty = TRUE` to opt out. That default caused unnecessary friction when a selector legitimately resolves to nothing (e.g. no nominal predictors in a dataset).

---

## `recipe.summary()`

Returns the current schema — see [schema.md](schema.md).
