# Selectors

→ See also: [roles.md](roles.md), [schema.md](schema.md), [step-contract.md](step-contract.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Selectors are a small expression DSL that resolves to a list of column names **at fit time**, using the schema after all preceding steps. They are predicate-based and composable with standard set operations.

Selectors work **everywhere a column name or list of column names is accepted** — not just in step arguments. This includes:

- Every step argument that names columns (not just the leading variadic)
- `set_roles(target=..., feature=...)`
- `update_role_requirements(role=..., ...)`
- `add_role()`, `update_role()`, `remove_role()`
- The column-subsetting argument of `transform()`

R `recipes` only supported selectors in the leading `...` of steps, making secondary-column args and role management impossible to express with the DSL [#1197, #1225, #1299, #1417]. We treat selectors as a first-class primitive that the entire API is built on top of.

---

The native DSL is exposed as factory methods on the `Selectors` namespace, conventionally imported as `from shoez import Selectors as s`. Every factory returns a `ShoezSelector` that composes with `&` / `|` / `~` / `-` and resolves to column names at fit time. The examples below use that `s.` prefix, which is the house style for all shoez examples; `s` also stays out of the way of `cs`, the conventional alias for `polars.selectors`.

## By name

```python
s.col("x1")                    # literal column name(s)
s.starts_with("x")
s.ends_with("_score")
s.contains("date")
s.matches(r"^x\d+$")           # regex (re.search)
s.num_range("x", range(1, 6))  # x1, x2, x3, x4, x5
s.everything()
```

`s.col` is **tolerant**: names not present in the frame are silently skipped (like R's `any_of`). Passing a plain `list` of names as an operand is **strict** and raises on a missing column (like `all_of`) — so `s.col("a", "missing")` is a no-op on the absent name, while `... & ["a", "missing"]` raises. Reach for `col` when absence should be a no-op, and a bare list when a missing name is a mistake worth surfacing.

There is deliberately **no positional range selector** (recipes' `beds:sqft`, selecting every column between two positions). Position is a brittle thing to depend on: it changes as upstream steps add and drop columns, so a range that is correct when the recipe is written can silently capture the wrong columns at fit time. Select by name pattern (`starts_with`, `matches`, `num_range`), by role, or by type instead.

---

## By role

```python
s.has_role("group")
s.features()
s.targets()
```

`s.features()` and `s.targets()` are exact aliases for `has_role("feature")` and `has_role("target")` — they exist for readability and as the conventional spelling (mirroring R recipes' `all_predictors()` / `all_outcomes()`), not because they apply any extra filtering. The target is excluded from `features()` purely because `target` and `feature` are mutually exclusive roles (see [`roles.md`](roles.md)); no column can carry both, so the bare role query already does the right thing.

---

## By source (producing step)

```python
s.has_source("dummy")               # columns whose immediate producer is this step id
s.has_source(step_type=StepDummy)   # columns produced directly by any StepDummy step
```

Because every step declares its input/output mode and the recipe maintains a column-lineage DAG, it is possible to select columns by *which step produced them* — not just by name, role, or type. This is directly useful for things like selecting all dummy-encoded columns to inspect or further transform them:

```python
rec = (
    Recipe(data=df)
    .add(StepDummy(s.categorical_features()))
    .add(StepPCA(s.has_source(step_type=StepDummy)))  # PCA only over the dummies
)
```

This was impossible in R `recipes` because provenance was not tracked [tidymodels/recipes#571]. It falls out naturally from the mode-declaration design in [step-contract.md](step-contract.md). The provenance selectors resolve against the lineage DAG at fit time — see [schema.md](schema.md).

### Direct vs. ancestral provenance — `has_source` vs `has_ancestor`

The lineage DAG distinguishes between a column whose **immediate** parent is a given step, and a column that has that step **anywhere in its ancestry** — e.g. a dummy column that was subsequently passed through `StepNormalize`. Both are useful, and shoez exposes them as **two separate selectors** (the decision for issue #20's "Q3"):

- **`has_source(...)` — direct:** "give me the columns that `StepDummy` just produced" — the raw dummies before any further transformation. Resolves via `Schema.columns_from_source`.
- **`has_ancestor(...)` — ancestral:** "give me all columns that trace back to `StepDummy` at any point" — dummies even after they've been normalized, scaled, or otherwise transformed. Resolves by walking `Schema.ancestors`.

Both accept either a step id (positional) or a step class (`step_type=`). Unlike the dtype and role selectors, neither is a silent no-op when nothing matches: an id or type that names no step earlier in the recipe raises `ValueError` (issue #220). The empty-selection-is-identity rule is right for a selector whose match set depends on the data (a frame with no numeric features), but a step id or type either names a step in the recipe or it is a typo, and that is knowable from the recipe alone. The special id `"original"` is always valid: it names the input columns that survive to that point. Separate functions were chosen over a `direct=`/`depth=` flag because the two intents read clearly at the call site and map one-to-one onto the two distinct DAG queries.

---

## By type

```python
s.numeric()        # float + int
s.float()          # alias: s.double  (R: all_double)
s.integer()        # (R: all_integer)
s.categorical()    # ordered + unordered (R: all_factor)
s.ordered()
s.unordered()
s.nominal()        # string + categorical + enum
s.string()         # (R: all_string)
s.boolean()        # (R: all_logical)
s.date()
s.datetime()
```

Names follow Python and polars conventions, not R. The dtype check is an internal primitive (see `shoez._frame`) and is not part of the public surface. See [schema.md](schema.md) for the full type taxonomy.

---

## Compound selectors

```python
s.numeric_features()       # numeric() & features()
s.double_features()        # float() & features()
s.integer_features()       # integer() & features()
s.categorical_features()   # categorical() & features()
s.string_features()        # string() & features()
s.nominal_features()       # nominal() & features()
```

These compound selectors are the **preferred API** for step arguments. They are clearer in intent and less error-prone than manually composing type and role selectors with `&`. The bare type selectors (`s.numeric()`, `s.float()`, etc.) exist for advanced use — for example inside role management calls — but in step arguments `s.numeric_features()` should nearly always be the right choice over `s.numeric()`.

---

## Set operations

All selectors compose freely with `&`, `|`, and `~`:

```python
s.features() & s.numeric()                       # intersection
s.features() | s.col("weight")                   # union
s.features() & ~s.targets()                     # difference
s.features() - s.targets()                      # difference (sugar)
s.has_source(step_type=StepDummy) & s.float()      # dummy columns that are float
s.has_source(step_type=StepDummy) | s.categorical()  # dummies or categoricals
~s.has_source(step_type=StepPCA)                           # everything except PCA outputs
```

Negation uses `~`; `-` is sugar for difference (`a - b` ≡ `a & ~b`). Set operations preserve the frame's column order regardless of operand order. Multiple positional arguments to a step are treated as a union.

---

## Evaluation timing

Selectors are resolved against the **post-step-N-1** schema at fit time, so a step that adds or removes columns affects what later selectors see. The recipe maintains a column-lineage DAG (see [schema.md](schema.md)) that selectors resolve against.

At pre-fit time, `repr(recipe)` shows the **raw selector expressions** (unresolved). After fit, it shows the **resolved column names**. See [inspection.md](inspection.md).

---

## Relationship to polars / narwhals selectors

The native `Selectors` DSL is the recommended surface: it covers name, type, role, and provenance in **one family** that always composes with `&` / `|` / `~` / `-`, and resolves against the lineage DAG.

`shoez` also still **accepts** a raw [polars selector](https://docs.pola.rs/api/python/stable/reference/selectors.html) (`polars.selectors` / `cs`) or narwhals selector (`narwhals.selectors`) anywhere a selector is expected — they are resolved against the frame schema at fit time, for any backend. This is a kept **escape hatch** for users who already think in polars, not the primary API:

```python
import polars.selectors as cs

StepNormalize(cs.numeric())   # accepted; resolves fine on pandas- or polars-backed frames
```

### Cross-family composition has a hard limit

polars and narwhals selectors compose cleanly *within their own family*, and native shoez selectors compose cleanly with each other. **Mixing the two families with `&` / `|` / `~` is asymmetric and only works when a shoez selector leads**, because shoez does not control polars'/narwhals' operators:

| Expression | Behavior |
| --- | --- |
| shoez & shoez | ✅ native composition |
| polars & polars, narwhals & narwhals | ✅ their own native composition |
| `shoez_sel & cs.numeric()` (shoez leads) | ✅ supported — shoez coerces the right operand |
| `cs.numeric() & shoez_sel` (polars leads) | ❌ polars raises `TypeError` |
| `ncs.numeric() & shoez_sel` (narwhals leads) | ❌ narwhals **silently** returns a broken `Expr` (no error) |

The narwhals case is the dangerous one: it produces no error at all, just a meaningless `Expr` that fails downstream. **Rule: a shoez-native selector must lead any cross-family expression.** To put a polars/narwhals selector first, lift it into shoez-land with `s.wrap`:

```python
s.wrap(cs.starts_with("x_")) & s.features()
```

In practice the native DSL covers the name/type surface, so cross-family mixing is rarely needed — but when it is, lead with shoez or use `wrap`.

---

## What to avoid

- No `*_at` / `*_if` family — R added these and they were superseded immediately by tidyselect [#662, #707].
- No data-value-dependent selectors (e.g. "select columns with missing values") — selectors are about identity, not data content [#617].
- No formula interface at all. We considered formulas as syntactic sugar over role assignment but decided against shipping them: roles are assigned explicitly instead. (Long formulas blew R's C stack [#1279, #1289, #1517] and `y ~ .` silently ignored `-` [#388].)
