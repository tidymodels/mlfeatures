# Lifecycle API

→ See also: [roles.md](roles.md), [selectors.md](selectors.md), [schema.md](schema.md), [serialization.md](serialization.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Minimum public surface

```python
rec = Recipe(data=df_train).set_roles(target=["y"], feature=["x1", "x2", ...])

rec = (
    rec
    .add(ImputeKNN(s.features()))
    .add(Dummy(s.nominal_features()))
    .add(Normalize(s.numeric_features()))
)

trained = rec.fit(df_train)        # fits statistics
X_train = trained.transform(df_train)    # applies to training data
X_test  = trained.transform(df_test)     # applies to held-out data
X_new   = trained.transform(df_new, s.features())  # selector at transform time
```

---

## Required behaviors

- `Recipe(...)` takes explicit role assignment from an example dataframe (used only for schema, not for fitting). There is **one canonical constructor**: roles are assigned explicitly via constructor kwargs or `set_roles`. **We have decided not to ship a formula interface.** It was originally planned as syntactic sugar over role assignment (simple `"y ~ ."` / `"y ~ x1 + x2"` splits only), but we removed it: it adds a parsing surface and a second way to do one thing, while explicit role assignment covers every case it could. Anything beyond basic column selection must be expressed as a step regardless.
- Steps are pure: `.add(Step(...))` returns a new `Recipe` with the step appended; it does **not** touch data.
- `fit()` runs each step's fit on the **training** data, in order. The transformed output of step *i* becomes the input to step *i+1*'s fit.
- `transform()` runs each step's transform on **new** data using stored statistics. `transform()` **never re-estimates**.
- `transform()` accepts an optional selector (see [selectors.md](selectors.md)) to subset returned columns.
- `transform()` always requires data to be passed in explicitly. There is no argument-free shorthand — callers are responsible for passing the dataframe they want transformed.
- `fit()` and `transform()` are idempotent given the same inputs.
- A trained recipe must support `recipe.minimize()` to strip training artifacts. See [serialization.md](serialization.md).

---

## Input validation via ptype snapshot

At `fit()` time a snapshot of sufficient schema information is captured from the training data and stored on the resulting `FittedRecipe` (as `FittedRecipe.ptype`):

- Column names
- Column dtypes
- Role assignments
- Per-column transform-time requirements (precomputed from the table below)

The snapshot is taken at `fit()` rather than construction so that role-defaulting has already run and the per-role requirements are resolved against the actual training columns. No data values are retained — only the structural description (the name→dtype half reuses narwhals' `Schema`). The snapshot is lightweight and survives serialization, so a pickled `FittedRecipe` keeps validating new data at the boundary.

### Per-role requirements

Whether a column must be present at `fit()` and `transform()` time is determined per-role. Some roles have hardcoded, unchangeable requirements; others have defaults the user can override:

| Role             | Required at `fit()` | Required at `transform()` | Configurable? |
| ---------------- | ------------------- | ------------------------- | ------------- |
| `"target"`      | yes                 | **no** (hardcoded)        | no            |
| `"feature"`    | yes                 | **yes** (hardcoded)       | no            |
| `"case_weights"` | yes                 | no (default)              | yes           |
| no role (NA)     | yes                 | yes (default)             | yes           |
| custom roles     | yes                 | yes (default)             | yes           |

All columns are required at `fit()` time regardless of role. Only `transform()`-time requirements vary.

User-configurable requirements are set via:

```python
# "id" is a user-defined custom role, not a built-in
recipe.update_role_requirements("id", transform_required=False)
# selectors work here too:
recipe.update_role_requirements(starts_with("id_"), transform_required=False)
```

Selector expressions are accepted wherever a column name is — see [selectors.md](selectors.md).

Unlike R's `update_role_requirements()`, this applies to **any** column in the schema — including columns produced by earlier steps — since requirements are tracked in the column-lineage DAG. See [schema.md](schema.md).

### What is checked

- All required columns are present (hard error if missing).
- Column dtypes match the ptype snapshot (hard error on mismatch).
- Extra columns beyond those in the ptype are silently ignored.

This catches the most common failure modes — passing the wrong dataframe, a column silently changing type between training and serving — at the boundary before any step runs.

---

## Row-altering steps and `skip`

Steps that alter rows (filter, sample, arrange) must **not** run on prediction-time data. Rather than a convention-based `skip=True` flag — which R `recipes` had to retrofit and which was routinely missed by step authors — row-altering steps are a **structural subclass**:

```python
class RowOp(Step):
    """Steps of this type are skipped at transform time automatically."""
```

This makes the skip behavior structural, not a boolean attribute. See [step-contract.md](step-contract.md) for the full step contract.

---

## What the framework handles — not individual steps

A significant design goal is that step authors should only need to implement the statistical logic of their step. Everything else is handled once, centrally, by the framework. In R `recipes`, each of the following had to be implemented (or forgotten) individually inside every step:

| Concern | R `recipes` | `shoez` |
| --------------------------------- | ------------------------------------ | ------------------------------- |
| Column presence at transform time | Each step checked, or silently broke | Ptype check before any step runs |
| Dtype validation | Each step did its own or skipped it | `accepts_dtypes()` checked by framework before `fit()` |
| Selector resolution | Each step called tidyselect internals | Framework resolves before calling `step.fit()` |
| Skip on new data (row ops) | `skip=TRUE` convention, routinely missed | Structural — `RowOp` subclass, framework enforces |
| Error messages with step identity | Each step wrapped errors, or didn't | Framework wraps all `fit()`/`transform()` calls |
| `keep_original_cols` | Implemented separately in each step, with an arbitrary per-step interface | Implemented in each step, but the contract fixes which modes expose it and what it defaults to |
| Output column name uniqueness | Each step checked, or collisions happened silently | Framework validates `produced_columns()` before each step runs |
| Lineage DAG update | Not tracked at all | Framework updates from `schema_out()` after each step |
| Sparse propagation | Each step handled or silently densified | Framework tracks through lineage DAG |

The result is that writing a new step is mostly writing `fit()` and `transform()` — the statistical core — and declaring a handful of properties (`accepts_dtypes`, `mode`, `output_role`). The framework does the rest.

---

## Internal mechanics

A rough sketch of what `fit()` and `transform()` do internally. This is not final implementation — it is a guide for what the implementation needs to accomplish.

### `recipe.fit(data)`

```python
def fit(self, data):
    # 1. snapshot the training schema into a ptype, then validate the training
    #    frame against it — all columns required at fit time. The snapshot is
    #    captured here (not at construction), so role-defaulting has run and the
    #    per-role transform requirements are known.
    ptype = Ptype.from_frame(data, roles, overrides)
    ptype.check(data, context="fit")

    # 2. iterate over steps in order
    current_data = data
    fitted_steps = []

    for step in self.steps:
        # resolve selectors against the current schema
        resolved_cols = step.selector.resolve(self._schema)

        # check that selected columns match the step's accepted dtypes
        check_accepted_dtypes(step, current_data, resolved_cols)

        # fit the step on the current data
        fitted_step = step.fit(current_data, info=self._schema)

        # update the lineage DAG with this step's schema_out()
        self._schema.update(fitted_step.schema_out())

        # for non-RowOp steps: transform current data to feed the next step
        if not isinstance(step, RowOp):
            current_data = fitted_step.transform(current_data)

        fitted_steps.append(fitted_step)

    return FittedRecipe(steps=fitted_steps, schema=self._schema, ptype=ptype)
```

Key points:
- The transformed output of step *i* is the input to step *i+1*'s `fit()` — so later steps see the schema as modified by earlier steps.
- `RowOp` steps are fitted (they may need to learn something, e.g. `step_filter` needs its condition) but their `transform()` is not called in the fit loop — row operations only apply to training data.
- The lineage DAG is updated after each step using `schema_out()`, so selector resolution at step *i+1* sees the correct post-step-*i* schema.

### `recipe.transform(data)`

```python
def transform(self, data, selector=None):
    # 1. validate input against the ptype captured at fit() — only columns
    #    flagged required at transform time (features always; targets never;
    #    others per role default / update_role_requirements) must be present.
    self.ptype.check(data, context="transform")

    # 2. apply each step in order, skipping RowOps
    current_data = data

    for step in self.fitted_steps:
        if isinstance(step, RowOp):
            continue  # never run on new data
        current_data = step.transform(current_data)

    # 3. optionally subset output columns by selector
    if selector is not None:
        output_cols = selector.resolve(self._schema)
        current_data = current_data.select(output_cols)

    return current_data
```

Key points:
- `RowOp` steps are skipped entirely — not fitted again, not transformed.
- No statistics are re-estimated. Each step uses only what was stored during `fit()`.
- The optional output selector is applied at the end, after all steps have run.

---

## Lazy vs. eager evaluation

Open question — see [goals.md § open questions](goals.md). Decision must be made explicitly before v0.1; R `recipes` never decided and paid for it. Recommendation: `transform()` returns an eager polars frame by default; a `transform_lazy()` variant can be added later.
