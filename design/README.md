# Design: `shoez`

A Python port of the R `recipes` package — a framework for **declaring** a pipeline of feature-engineering operations, **estimating** any required statistics from a training set, and **applying** those operations identically to new data.

---

## What `recipes` is

The framework separates three lifecycle phases:

1. **Specify** — declare a sequence of steps against a *schema* of variables (names, types, roles). No computation on data yet.
2. **Fit** — fit each step in order on a training dataset. Statistics (means, factor levels, PCA loadings, …) are estimated and stored inside the recipe.
3. **Transform** — apply the fitted recipe to any dataset (train, test, or new prediction-time data) and return a transformed table.

Key invariants the framework guarantees:

- **Train/serve parity.** The same statistics are used for training and inference. Test data never leaks into estimation.
- **Deferred evaluation.** Constructing a recipe and adding steps (`Recipe(steps=[...])` / `.add(...)`) declares intent against a schema — no data is touched. All computation is deferred to `fit()` and `transform()`. This means recipes are cheap to construct, inspect, and modify before any data flows through them.
- **Composability.** Steps are independent objects appended to a recipe; order is explicit and meaningful.
- **Schema awareness.** Each step selects its inputs by **name**, **role**, or **type** — not by positional index.
- **Extensibility.** Third-party packages can contribute new steps, checks, and roles without modifying core. The plugin contract is stable and documented; see [extensibility.md](extensibility.md).

---

## Core concepts

| Concept            | R term      | Meaning                                                                                                                                                                          |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Column**         | variable / term | A column, whether in the input table or the output design matrix. A derived column (e.g. a dummy or PCA component) is as much a column as a raw input one. |
| **Sample**         | observation / case | One row of the data. |
| **Feature**        | predictor   | A column used as model input, i.e. one carrying the `"feature"` role. Not a synonym for "column": a target or an ID column is a column but not a feature. |
| **Target**         | outcome     | The column being predicted, i.e. one carrying the `"target"` role. |
| **Role**           | role        | A user-assigned label describing *how* a column is used: `"feature"`, `"target"`, `"id"`, `"case_weights"`, or any user-defined string. A column may have multiple roles. |
| **Type**           | type        | Inferred data type with a hierarchy: `double`, `int`, `bool`, `string`, `factor[ordered]`, `factor[unordered]`, `date`, `datetime`.                                              |
| **Selector**       | selector    | A small DSL for choosing columns by name, role, or type.                                                                                                                       |
| **Step**           | `step_*()`  | A single transformation (e.g. center, dummy-encode, PCA).                                                                                                                        |
| **Recipe**         | `recipe`    | The ordered container of steps + a schema describing input columns.                                                                                                            |

---

## Document index

| File                                   | Covers                                               |
| -------------------------------------- | ---------------------------------------------------- |
| [motivation.md](motivation.md)         | Worked example: why not just a sklearn Pipeline      |
| [lifecycle.md](lifecycle.md)           | The fit/transform API and lifecycle guarantees       |
| [interfaces.md](interfaces.md)         | Construction syntax — step objects and the `>>` pipe |
| [roles.md](roles.md)                   | Role system and role management API                  |
| [selectors.md](selectors.md)           | Selector DSL — by name, role, type, and set ops      |
| [steps-catalog.md](steps-catalog.md)   | Full step catalog, tiered by priority                |
| [step-contract.md](step-contract.md)   | What every step must implement                       |
| [schema.md](schema.md)                 | Column-lineage DAG and schema tracking               |
| [ecosystem.md](ecosystem.md)           | Dataframe types, sparse support, sklearn adapter     |
| [serialization.md](serialization.md)   | Pickle + stable format, `minimize()`, no closures    |
| [extensibility.md](extensibility.md)   | Plugin steps via entry points, base classes          |
| [inspection.md](inspection.md)         | `repr`, errors, `summary()`                          |
| [dependencies.md](dependencies.md)     | Optional dependencies — lazy imports in steps        |
| [goals.md](goals.md)                   | Non-goals, open questions, v0.1 success criteria     |
| [naming_conventions.md](naming_conventions.md) | Terminology and the `step_drop_*` prefix rule |
