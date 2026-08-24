# Roles

→ See also: [selectors.md](selectors.md), [schema.md](schema.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Roles are a first-class concept and are what makes `shoez` different from a typical sklearn `Pipeline`. Every column in the recipe carries **zero or more** role names stored as a `set[str]`.

---

## Standard roles

| Role             | Meaning                                         |
| ---------------- | ----------------------------------------------- |
| `"feature"`    | Used as a model input                           |
| `"target"`      | The column(s) to predict                        |
| `"case_weights"` | Per-sample weights |

Users may add arbitrary string roles (`"id"`, `"group"`, `"date_split"`, etc.). New roles register cheaply — no coordinated changes across packages required.

---

## Multi-role variables

A column may carry multiple roles simultaneously. Validation **rejects nonsensical combinations** at assignment time: a column cannot be both `"target"` and `"feature"`.

```python
# A timestamp column used both as a feature and for time-based splitting
rec = Recipe(data=df).set_roles(
    target=["price"],
    feature=["sqft", "timestamp"],
).add_role("timestamp", role="date_split")
# timestamp now has roles {"feature", "date_split"}

# A weight column used both for model weighting and as a feature
rec = rec.add_role("sample_weight", role="case_weights")
# sample_weight now has roles {"feature", "case_weights"}

# Survival analysis: both time and event columns are targets
rec = Recipe(data=df).set_roles(
    target=["surv_time", "event"],
    feature=["age", "treatment"],
)
# surv_time and event each have role {"target"}
```

A column with no role is retained in the schema but is not selected by `s.features()` etc.

Targets are a structured object (or a role applied to multiple columns), not a single formula LHS. This generalizes cleanly to multi-output regression, survival, and multitask learning. See [goals.md § open questions](goals.md).

---

## Role management API

```python
recipe.add_role("col", role="group")
recipe.update_role("col", new_role="id", old_role="feature")
recipe.remove_role("col", role="group")
```

### Inline with steps — a key design priority

Role management calls are **first-class pipeline operations**, interleaved with steps in whatever order makes sense:

```python
rec = (
    Recipe(data=df)
    .add(ImputeKNN(s.features()))
    .add_role(starts_with("key_"), role="group")   # role change mid-pipeline
    .add(Dummy(s.nominal_features()))
    .update_role("weight", new_role="case_weights")
    .add(Normalize(s.numeric_features()))
)
```

In R `recipes`, role management is technically applied to the original data before any steps run, regardless of where it appears in the code. This caused persistent confusion — users expected role changes to take effect at the point they wrote them, and were surprised when they didn't [#437, #778]. In `shoez`, the order is respected: a role change takes effect from that point in the pipeline forward, affecting which columns subsequent selectors resolve to.

Role management also operates on **any** column in the schema — including columns produced by earlier steps — since the lineage DAG is updated after each operation. See [schema.md](schema.md).

---

## Roles on produced columns

Steps that create new columns must declare a `role=` for those columns (default `"feature"`). The user may override `role=` at the step call site:

```python
recipe.add(ClassDist(s.features(), role="distance"))
```

---

## Roles vs. flags

When a step "feels metadata-ish," it probably should be a role instead. The R `recipes` `step_protect` experiment [#252] proved this — it was abandoned in favour of a role. If a proposed step primarily controls *how a column is treated downstream* rather than *transforming values*, use a role.
