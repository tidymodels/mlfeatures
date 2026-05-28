# Roles

→ See also: [selectors.md](selectors.md), [schema.md](schema.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Roles are a first-class concept and are what makes `pyrecipes` different from a typical sklearn `Pipeline`. Every variable in the recipe carries **zero or more** role labels stored as a `set[str]`.

---

## Standard roles

| Role             | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| `"predictor"`    | Used as a model input feature                   |
| `"outcome"`      | The target variable(s) to predict               |
| `"case_weights"` | Per-row observation weights |

Users may add arbitrary string roles (`"id"`, `"group"`, `"date_split"`, etc.). New roles register cheaply — no coordinated changes across packages required.

---

## Multi-role variables

A variable may carry multiple roles simultaneously. Validation **rejects nonsensical combinations** at assignment time — e.g. a variable cannot be both `"outcome"` and `"predictor"`.

```python
# A timestamp column used both as a predictor feature and for time-based splitting
rec = Recipe(data=df).set_roles(
    outcome=["price"],
    predictor=["sqft", "timestamp"],
).add_role("timestamp", role="date_split")
# timestamp now has roles {"predictor", "date_split"}

# A weight column used both for model weighting and as a predictor
rec = rec.add_role("sample_weight", role="case_weights")
# sample_weight now has roles {"predictor", "case_weights"}

# Survival analysis: both time and event columns are outcomes
rec = Recipe(data=df).set_roles(
    outcome=["surv_time", "event"],
    predictor=["age", "treatment"],
)
# surv_time and event each have role {"outcome"}
```

A variable with no role is retained in the schema but is not selected by `all_predictors()` etc.

Outcomes are a structured object (or a role applied to multiple columns), not a single formula LHS. This generalizes cleanly to multi-output regression, survival, and multitask learning. See [goals.md § open questions](goals.md).

---

## Role management API

```python
recipe.add_role("col", role="group")
recipe.update_role("col", new_role="id", old_role="predictor")
recipe.remove_role("col", role="group")
```

### Inline with steps — a key design priority

Role management calls are **first-class pipeline operations**, interleaved with steps in whatever order makes sense:

```python
rec = (
    Recipe(data=df)
    .step_impute_knn(all_predictors())
    .add_role(starts_with("key_"), role="group")   # role change mid-pipeline
    .step_dummy(all_nominal_predictors())
    .update_role("weight", new_role="case_weights")
    .step_normalize(all_numeric_predictors())
)
```

In R `recipes`, role management is technically applied to the original data before any steps run, regardless of where it appears in the code. This caused persistent confusion — users expected role changes to take effect at the point they wrote them, and were surprised when they didn't [#437, #778]. In `pyrecipes`, the order is respected: a role change takes effect from that point in the pipeline forward, affecting which columns subsequent selectors resolve to.

Role management also operates on **any** column in the schema — including columns produced by earlier steps — since the lineage DAG is updated after each operation. See [schema.md](schema.md).

---

## Roles on produced columns

Steps that create new columns must declare a `role=` for those columns (default `"predictor"`). The user may override `role=` at the step call site:

```python
recipe.step_classdist(all_predictors(), role="distance")
```

---

## Roles vs. flags

When a step "feels metadata-ish," it probably should be a role instead. The R `recipes` `step_protect` experiment [#252] proved this — it was abandoned in favour of a role. If a proposed step primarily controls *how a column is treated downstream* rather than *transforming values*, use a role.
