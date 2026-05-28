# Schema Tracking

→ See also: [roles.md](roles.md), [selectors.md](selectors.md), [step-contract.md](step-contract.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Column-lineage DAG

The recipe maintains an internal **column-lineage DAG** (not a flat table) that is updated after each step's `fit()`. R `recipes` used a flat `var_info` / `term_info` table, which made provenance impossible and caused years of issues [#571, #604, #611, #866].

Per column node:

| Field           | Type                      | Notes                                    |
| --------------- | ------------------------- | ---------------------------------------- |
| `name`          | `str`                     | Current column name                      |
| `dtype`         | `DType`                   | Current dtype (may change across steps)  |
| `roles`         | `set[str]`                | Current roles — see [roles.md](roles.md) |
| `source`        | `"original" \| step_id`   | Which step produced this column          |
| `parent_cols`   | `list[str]`               | Input columns that produced this one     |
| `dtype_history` | `list[DType]`             | Dtype at each step boundary              |
| `sparse`        | `bool`                    | Whether this column is currently sparse  |

This DAG enables:
- `has_source(step_id)` selectors.
- Sparse propagation tracking — see [ecosystem.md](ecosystem.md).
- Densification warnings when a downstream step would destroy sparse columns.
- Consistent inspection, since provenance is in the DAG.

---

## Type taxonomy

Richer than R's nominal/numeric binary [#942, #602, #63], and named after Python/polars conventions rather than R:

```
numeric
  float      (R: double)
  int
bool         (R: logical)
str
categorical
  ordered
  unordered  (R: factor)
date
datetime
duration
list[<type>]
```

Each step declares which dtypes it `accepts_dtypes()` — rejection happens at the schema boundary, not buried in fitting code. This catches type mismatches before any data is processed.

---

## `recipe.summary()`

Returns the current schema as a tidy `DataFrame` with columns: `name`, `dtype`, `roles`, `source`, `sparse`. Before fit, shows the initial schema. After fit, shows the post-all-steps schema.

---

## Schema at pre- and post-fit

- **Pre-fit:** selectors are shown as unresolved expressions.
- **Post-fit:** selectors show resolved column names; the DAG is fully populated.

See [inspection.md](inspection.md) for how this feeds into `repr(recipe)`.
