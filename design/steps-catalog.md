# Steps Catalog

→ See also: [step-contract.md](step-contract.md), [extensibility.md](extensibility.md)

> **Note:** Step names are not final and will need to be reconsidered before v0.1.

The first prototype needs exactly enough steps to exercise every input/output mode (see [step-contract.md § input/output mode](step-contract.md)) and to demonstrate steps that work with different input types. A broader catalog comes later.

---

## Prototype step set

One representative per mode, chosen to also cover numeric and categorical input types:

| Step | Mode | Input type | What it does |
| ------------------- | -------------- | ----------- | ------------------------------------------------- |
| `step_normalize` | `ONE_TO_ONE` | numeric | Z-score each column |
| `step_unknown` | `ONE_TO_ONE` | categorical | Replace NA levels with an explicit "unknown" level |
| `step_dummy` | `ONE_TO_MANY` | categorical | One-hot encode categorical columns |
| `step_poly` | `ONE_TO_MANY` | numeric | Expand a column into polynomial features |
| `step_pca` | `MANY_TO_MANY` | numeric | Project numeric columns into PCA components |
| `step_zv` | `REMOVING` | any | Drop zero-variance columns |
| `step_intercept` | `ADDING` | — | Add a constant column of ones |
| `step_filter` | `ROW_OP` | — | Remove rows matching a condition |
| `step_mutate` | `UNCONSTRAINED` | any | Arbitrary column mutation (escape hatch) |

This set is sufficient to build and validate the full framework — the lineage DAG, selector resolution, ptype checking, serialization, and the sklearn adapter — before expanding the catalog.
