# Selectors

→ See also: [roles.md](roles.md), [schema.md](schema.md), [step-contract.md](step-contract.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Selectors are a small expression DSL that resolves to a list of column names **at fit time**, using the schema after all preceding steps. They are predicate-based and composable with standard set operations.

Selectors work **everywhere a column name or list of column names is accepted** — not just in step arguments. This includes:

- Every step argument that names columns (not just the leading variadic)
- `set_roles(outcome=..., predictor=...)`
- `update_role_requirements(role=..., ...)`
- `add_role()`, `update_role()`, `remove_role()`
- The column-subsetting argument of `transform()`

R `recipes` only supported selectors in the leading `...` of steps, making secondary-column args and role management impossible to express with the DSL [#1197, #1225, #1299, #1417]. We treat selectors as a first-class primitive that the entire API is built on top of.

---

## By name

```python
col("x1")                    # literal column name
starts_with("x")
ends_with("_score")
contains("date")
matches(r"^x\d+$")           # regex
num_range("x", range(1, 6))  # x1, x2, x3, x4, x5
everything()
```

---

## By role

```python
has_role("group")
all_predictors()
all_outcomes()
```

Shorthands like `all_predictors()` exist because `has_role("predictor")` would otherwise inadvertently include outcome columns when used naively.

---

## By source (producing step)

```python
has_source("dummy_abc123")          # all columns produced by a specific step id
has_source(step_type=step_dummy)    # all columns produced by any step_dummy step
```

Because every step declares its input/output mode and the recipe maintains a column-lineage DAG, it is possible to select columns by *which step produced them* — not just by name, role, or type. This is directly useful for things like selecting all dummy-encoded columns to inspect or further transform them:

```python
rec = (
    Recipe(data=df)
    .add(Dummy(all_categorical_predictors()))
    .add(PCA(has_source(step_type=step_dummy)))  # PCA only over the dummies
)
```

This was impossible in R `recipes` because provenance was not tracked [tidymodels/recipes#571]. It falls out naturally from the mode-declaration design in [step-contract.md](step-contract.md). The `has_source()` selector resolves against the lineage DAG at fit time — see [schema.md](schema.md).

### Direct vs. ancestral provenance

The lineage DAG distinguishes between a column whose **immediate** parent is a given step, and a column that has that step **anywhere in its ancestry** — e.g. a dummy column that was subsequently passed through `step_normalize`. Both are useful:

- **Direct:** "give me the columns that `step_dummy` just produced" — the raw dummies before any further transformation.
- **Ancestral:** "give me all columns that trace back to `step_dummy` at any point" — dummies even after they've been normalized, scaled, or otherwise transformed.

The exact API for expressing this distinction is not yet decided — possible forms include a `direct=True` flag, separate functions (`has_source` vs `has_ancestor`), or a `depth=` argument. This should be resolved before v0.1.

---

## By type

```python
all_numeric()          # float + int
all_float()            # (R: all_double)
all_int()              # (R: all_integer)
all_categorical()      # ordered + unordered (R: all_factor)
all_ordered()
all_unordered()
all_str()              # (R: all_string)
all_bool()             # (R: all_logical)
all_date()
all_datetime()
```

Names follow Python and polars conventions, not R. `has_type()` is an internal primitive used to implement these and is not part of the public surface. See [schema.md](schema.md) for the full type taxonomy.

---

## Compound selectors

```python
all_numeric_predictors()       # all_numeric() & all_predictors()
all_float_predictors()         # all_float() & all_predictors()
all_int_predictors()           # all_int() & all_predictors()
all_categorical_predictors()   # all_categorical() & all_predictors()
all_str_predictors()           # all_str() & all_predictors()
```

These compound selectors are the **preferred API** for step arguments. They are clearer in intent and less error-prone than manually composing type and role selectors with `&`. The bare type selectors (`all_numeric()`, `all_float()`, etc.) exist for advanced use — for example inside role management calls — but in step arguments `all_numeric_predictors()` should nearly always be the right choice over `all_numeric()`.

---

## Set operations

All selectors compose freely with `&`, `|`, and `~`:

```python
all_predictors() & all_numeric()                        # intersection
all_predictors() | col("weight")                        # union
all_predictors() & ~all_outcomes()                      # difference
has_source(step_type=step_dummy) & all_float()          # dummy columns that are float
has_source(step_type=step_dummy) | all_categorical()    # dummies or categoricals
~has_source(step_type=step_pca)                         # everything except PCA outputs
```

Negation uses `~` (or `-` as sugar). Multiple positional arguments to a step are treated as a union.

---

## Evaluation timing

Selectors are resolved against the **post-step-N-1** schema at fit time, so a step that adds or removes columns affects what later selectors see. The recipe maintains a column-lineage DAG (see [schema.md](schema.md)) that selectors resolve against.

At pre-fit time, `repr(recipe)` shows the **raw selector expressions** (unresolved). After fit, it shows the **resolved column names**. See [inspection.md](inspection.md).

---

## Relationship to polars selectors

Since `shoez` uses polars internally, the [polars selectors API](https://docs.pola.rs/api/python/stable/reference/selectors.html) (`polars.selectors` / `cs`) is a natural reference point and potential foundation. Polars selectors already cover most of the by-type and by-name surface:

```python
import polars.selectors as cs

cs.numeric()        # ~ all_numeric()
cs.string()         # ~ all_string()
cs.by_name("x1")    # ~ col("x1")
cs.starts_with("x") # ~ starts_with("x")
cs.matches(r"...")   # ~ matches(r"...")
```

Polars selectors also compose with `&`, `|`, and `~` natively, which aligns exactly with our set-operation design.

The main gap is **role-based selection** — polars has no concept of roles, so `all_predictors()`, `all_outcomes()`, etc. will need to be layered on top. But for the type- and name-based selectors, building on or wrapping `polars.selectors` rather than reimplementing from scratch is worth serious consideration. It would reduce maintenance burden and give users a familiar API if they already know polars.

---

## What to avoid

- No `*_at` / `*_if` family — R added these and they were superseded immediately by tidyselect [#662, #707].
- No data-value-dependent selectors (e.g. "select columns with missing values") — selectors are about identity, not data content [#617].
- Formulas as selectors are limited to syntactic sugar over role assignment; do not make them the canonical entry point. Long formulas blew R's C stack [#1279, #1289, #1517] and `y ~ .` silently ignored `-` [#388].
