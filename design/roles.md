# Roles

→ See also: [selectors.md](selectors.md), [lineage.md](lineage.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Roles are a first-class concept and are what makes `mlfeatures` different from a typical sklearn `Pipeline`. Every column in the recipe carries **zero or more** role names stored as a `set[str]`.

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
    target="price",
    feature=["sqft", "timestamp"],
    date_split="timestamp",          # any extra keyword is a custom role
)
# timestamp now has roles {"feature", "date_split"}

# A weight column used both for model weighting and as a feature
rec = rec.set_roles(case_weights="sample_weight", feature="sample_weight")
# sample_weight now has roles {"feature", "case_weights"}

# Survival analysis: both time and event columns are targets
rec = Recipe(data=df).set_roles(
    target=["surv_time", "event"],
    feature=["age", "treatment"],
)
# surv_time and event each have role {"target"}
```

`set_roles()` only *adds* roles, and accepts column names, not selectors. A column that ends up with no role is retained in the schema but is not selected by `s.features()` etc.

**Default features.** If a recipe declares no `"feature"` column at all, fitting gives the role to every column that is not a target or a case-weights column. That includes a column that only has a custom role such as `"id"`, so declare `features` explicitly whenever such a column should stay out of the model inputs.

Targets are a structured object (or a role applied to multiple columns), not a single formula LHS. This generalizes cleanly to multi-output regression, survival, and multitask learning. See [goals.md § open questions](goals.md).

---

## Role management API

`Recipe.set_roles()` declares roles on the **input columns**, whatever its position among the steps. Three further methods change roles part way through the pipeline:

```python
recipe.add_role("col", role="group")
recipe.update_role("col", new_role="id", old_role="feature")
recipe.remove_role("col", role="group")
```

Each takes a column name, a list of names, or any selector, like a step's `columns`, and returns a new recipe. `update_role` without `old_role` replaces all of a column's roles; with it, only that role is swapped. `update_role(old_role=...)` and `remove_role` raise at fit time if a selected column does not carry the role. A selection matching nothing is a no-op, as for a step.

### Inline with steps

Role operations are recorded at their position in the chain, interleaved with steps:

```python
rec = (
    Recipe(data=df)
    .add(StepImputeMedian(s.numeric_features()))
    .add_role(s.starts_with("key_"), role="group")   # role change mid-pipeline
    .add(StepDummy(s.nominal_features()))
    .update_role("weight", new_role="case_weights")
    .add(StepNormalize(s.numeric_features()))
)
```

In R `recipes`, role management is technically applied to the original data before any steps run, regardless of where it appears in the code. This caused persistent confusion: users expected role changes to take effect at the point they wrote them, and were surprised when they didn't [#437, #778]. In `mlfeatures`, the order is respected: a role change takes effect from that point in the pipeline forward, affecting which columns subsequent selectors resolve to.

Role operations work on **any** column live at that point, including columns produced by earlier steps, and the lineage DAG's roles are refreshed after each one. See [lineage.md](lineage.md).

Implementation notes:

- A role operation is not a step: it has no id, is not in `Recipe.steps`, and is not part of `get_params()`. It is stored with the number of steps before it and applied by `fit()` between steps. Roles never change data, so `transform()` has nothing to replay, and the `FittedRecipe` keeps only the final roles.
- Selectors are resolved at fit time against the columns and roles as they stand at that position.
- Operations written before the first step (leading operations) run after feature defaulting and before the schema snapshot. So they read the same roles they would after a step, and a role selector such as `s.numeric_features()` sees the default features there too. One rule keeps a leading operation equal to the `set_roles()` declaration it stands in for, and it only concerns a feature role that defaulting handed out, never one the user declared: giving a defaulted column the `target` or `case_weights` role takes the default feature role away, because defaulting never hands those columns the role (`add_role("y", role="target")` matches `set_roles(target="y")`). Position does not matter here: written after a step, the same call settles a defaulted feature the same way (the fit carries the set of default features through every role operation), so declaring a target never needs a `set_roles()`.

  Naming the feature role is where the two differ. `set_roles(feature="x")` and the `features=` argument *choose* the features, so no other default survives them, and that holds at any position. `add_role("x", role="feature")` and `update_role(..., new_role="feature")` only give the role to `x`, as they do after a step, and leave the other default features alone. Position must not change what a role operation does, and the declaration is the call that says "these are the features". The user's own feature role on `x` is then no longer a default, so a later operation cannot quietly displace it.

  Defaulting happens once: if `features` was declared there is none, and an operation that takes a declared feature's role away does not trigger it again. The untrained `repr` and `summary()` run the same operations but leave out the default features that remain, so they show only what was declared.
- `set_roles()` always acts on the input columns. Once a leading operation exists, later `set_roles()` calls are recorded after it in written order (shown in the `repr` as `set_roles(...)`), so `update_role("x", new_role="id").set_roles(group="x")` leaves `x` with both roles. Before any leading operation it just adds to the declared roles, which is the same thing. `Recipe.roles` returns a copy of the roles the recipe starts from, with the leading operations applied (default features are left out), and the copy is read-only, so `recipe.roles.add(...)`, `recipe.roles.remove(...)` and assigning `recipe.roles` raise instead of silently changing (or failing to change) the recipe; use `set_roles()` or the role operations, or `recipe.roles.copy()` for a mutable copy (this replaces the earlier mutable `roles` attribute); `summary()` shows the same roles per column. A role selector returns columns in the data frame's column order, not in the order `Roles` holds them, which shifts as roles are removed and added.
- Later operations do not change `FittedRecipe.schema.roles`, which records the roles of the input columns as the data entered the recipe. They can change `FittedRecipe.schema.transform_required`, so that is the field to read for what `transform()` needs. For each input column such an operation touched, the requirement is derived again from its final roles, with the `update_role_requirements` overrides resolved against those roles (a column that gained a role takes that role's override). A column a step read while it was required stays required, otherwise a missing column would pass the schema check and fail inside the step; a row operation reads nothing at `transform()`, so it never counts. Columns no later operation touched keep the requirement they had at the snapshot.

---

## Roles on produced columns

Steps that create new columns must declare a `role=` for those columns (default `"feature"`). The user may override `role=` at the step call site:

```python
recipe.add(ClassDist(s.features(), role="distance"))
```

---

## Roles vs. flags

When a step "feels metadata-ish," it probably should be a role instead. The R `recipes` `step_protect` experiment [#252] proved this — it was abandoned in favour of a role. If a proposed step primarily controls *how a column is treated downstream* rather than *transforming values*, use a role.
