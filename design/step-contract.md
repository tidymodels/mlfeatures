# Step Contract

→ See also: [steps-catalog.md](steps-catalog.md), [schema.md](schema.md), [selectors.md](selectors.md), [extensibility.md](extensibility.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


The step contract is defined **before** any step is written, as an executable `StepConformanceTest` that every step must pass. R `recipes` discovered these requirements piecemeal over years of PRs [#215, #813, #993, #1167, #1329, #1472, #1486].

---

## Base class

```python
from abc import ABC, abstractmethod

class Step(ABC):
    id: str                      # content-derived, stable, user-overridable
    trained: bool
    keep_original_cols: bool
    output_role: str | None      # role assigned to new columns produced by this step

    @abstractmethod
    def fit(self, training_data: DataFrame, info: SchemaInfo) -> "Step": ...

    @abstractmethod
    def transform(self, new_data: DataFrame) -> DataFrame: ...

    @abstractmethod
    def schema_in(self) -> ColSchema: ...

    @abstractmethod
    def schema_out(self) -> ColSchema: ...

    @abstractmethod
    def accepts_dtypes(self) -> AcceptedDtypes: ...

    @property
    @abstractmethod
    def produces_sparse(self) -> bool: ...

    def required_pkgs(self) -> list[str]: return []


class RowOp(Step):
    """Row-altering steps. transform() is a no-op on new data — these steps only
    take effect during fit(). The no-op is enforced structurally by the framework's
    type-based dispatch, not by RowOp overriding transform()."""
```

There is no `skip` flag. Whether a step runs at transform time is determined entirely by its type: `Step` subclasses always run; `RowOp` subclasses never run on new data. This makes the behaviour structural and impossible to accidentally omit — the primary failure mode in R `recipes`, where `skip=TRUE` was a convention routinely missed by step authors [tidymodels/recipes#934, #1254].

---

## Constructor requirements

Every step constructor must accept:

| Argument                      | Type                | Notes                                                                                                          |
| ----------------------------- | ------------------- | -------------------------------------------------------------------------------------------------------------- |
| selector(s)                   | `Selector`          | Resolved at fit time against schema. Usable in **every** column-naming arg — see [selectors.md](selectors.md). |
| step-specific hyperparameters | varies              | Type-validated at construction time.                                                                           |
| `role`                        | `str \| None`       | Role assigned to *new* columns.                                                                                |
| `keep_original_cols`          | `bool`              | Whether to retain the source columns.                                                                          |
| `prefix` / `name_fn`          | `str \| callable`   | Controls generated column names.                                                                               |
| `id`                          | `str`               | Stable identifier; auto-generated if not provided.                                                             |

These are constructor arguments on the concrete step, not slots on the `Step` base class. The selector is stored as an instance attribute on the concrete step and resolved against the schema during `fit()` — the base class has no knowledge of which columns were selected.

All declared arguments are type-validated at construction. No silent coercions. Environment-dependent settings (locale, contrast schemes) are **captured at `fit()` time** and stored on the step — never read from global state at `transform()` time [#1349].

The stronger goal is to avoid environment-dependent settings entirely. Every behaviour that affects output should be an explicit constructor argument. Capturing from the environment at fit time is a fallback for cases where that is genuinely impractical — it should not be the default approach.

---

## `accepts_dtypes()`

Each step declares the set of column dtypes it is valid to apply to. The framework checks this against the schema **before `fit()` runs only** — if any selected column has a dtype not in the declared set, an error is raised immediately rather than failing deep inside the step's fitting code.

This check does not run at `transform()` time. By then, the ptype snapshot already guarantees that incoming column dtypes match what was seen during `fit()` — a second dtype check would be redundant. See [lifecycle.md](lifecycle.md).

The return type is `AcceptedDtypes`, an alias for `set[str] | _AnyDtype` — either an explicit set of dtype names, or the `ANY_DTYPE` sentinel. (A plain `set[str]` annotation would be inconsistent, since `ANY_DTYPE` is not a set.)

```python
# e.g. step_normalize only makes sense on numeric columns
def accepts_dtypes(self) -> AcceptedDtypes:
    return {"float", "int"}

# e.g. step_dummy only makes sense on categorical columns
def accepts_dtypes(self) -> AcceptedDtypes:
    return {"categorical"}

# e.g. step_zv works on anything — use the ANY sentinel
def accepts_dtypes(self) -> AcceptedDtypes:
    return ANY_DTYPE
```

`ANY_DTYPE` is a sentinel constant that skips dtype validation entirely for that step. It is the correct choice for steps like `step_zv` or `step_rm` where the operation is meaningful regardless of column type.

This replaces the R pattern of silently coercing or producing nonsense output when a step receives the wrong type. The check happens at the schema boundary — see [schema.md](schema.md) for the full type taxonomy — so the error message names the offending column and its actual dtype, not a cryptic failure from inside numpy or polars.

---

## Input/output lineage information

Each step must record what it consumed and what it produced after `fit()` runs. Getting this information before data flows through a step is not a goal — it is fine for it to be known only after fitting. What matters is that it is captured and available, so that:

- Later steps can resolve selectors against an accurate schema.
- `has_source()` selectors work after the recipe is fitted.
- Provenance can be traced — e.g. which input columns contributed to a PCA component or a set of dummy columns.

The mode labels below are a shorthand for the *shape* of the input/output relationship, which informs how the framework records lineage and what information a step needs to provide:

| Mode | Input/output relationship | Examples |
| -------------- | ------------------------------------------------------------ | --------------------------------------- |
| `ONE_TO_ONE` | Each input column produces exactly one output column | `step_normalize`, `step_impute_mean` |
| `ONE_TO_MANY` | Each input column produces one or more output columns | `step_dummy`, `step_poly` |
| `MANY_TO_MANY` | A set of input columns produces a new set of output columns | `step_pca`, `step_ica` |
| `REMOVING` | Input columns are dropped; no new columns created | `step_zv`, `step_drop_corr` |
| `ADDING` | New columns added with no corresponding input | `step_intercept` |
| `ROW_OP` | No column schema change | `step_filter`, `step_shuffle` |
| `UNCONSTRAINED` | Relationship cannot be declared in advance | `step_mutate` |

### What recording this information enables

After `fit()`, the lineage DAG contains the full input/output mapping for every step. This unlocks:

- **Feature provenance.** The lineage DAG records which input columns produced which output columns. Tracing PCA components or dummy columns back to their source is possible programmatically. This is the core problem identified in [tidymodels/recipes#1158](https://github.com/tidymodels/recipes/issues/1158).
- **`has_source()` selector.** Users can select all columns produced by a given step or step type — e.g. `has_source(step_type=step_dummy)`. See [selectors.md](selectors.md).
- **Naming contract enforcement.** The framework validates that generated column names are unique after each step runs.
- **`keep_original_cols` applicability.** Only meaningful for `ONE_TO_ONE` and `ONE_TO_MANY` modes — the framework ignores it for others.

---

## `fit()`

- Resolves selectors against the current schema.
- Estimates and stores statistics.
- Returns the fitted step. Setting `trained=True` is done by `recipe.fit()` after calling `step.fit()` — not by the step itself.
- Declares `schema_in()` and `schema_out()` so the recipe can update its column-lineage DAG. See [schema.md](schema.md).
- Must be idempotent given the same training data.

---

## `transform()`

- Applies stored statistics to `new_data`. Never re-estimates.
- Wrapped by the base class to add exception chaining: `raise StepError(step_id=self.id, step_type=type(self).__name__, columns=...) from underlying_exc`. See [inspection.md](inspection.md).
- Sparse-in → sparse-out (or explicit densification warning) where declared. See [ecosystem.md](ecosystem.md).

---

## `StepConformanceTest`

Every step must pass this test suite before merging:

- Empty selection returns empty output without error.
- 0-row and 1-row `transform()` produce schema-correct output.
- All declared constructor arguments are type-validated.
- `fit()` is idempotent given the same training data.
- `transform()` on training data after `fit()` reproduces `fit_transform()` output.
- Sparse-in → sparse-out (or explicit densification warning) where declared.
- Pickle round-trip preserves `transform()` output.
- `id` is stable across sessions for the same hyperparameters.

---

## Step IDs

Every step has a stable, content-derived `id` by default (e.g. `"normalize_1a2b3c"`), user-overridable. All inspection and tuning APIs key on `id`, never on position. Position-based indexing is deprecated from the start.
IDs must **not** use randomness. The suffix is derived deterministically from the step's type and hyperparameters — e.g. a hash of the constructor arguments. The same step type with the same arguments must produce the same `id` across sessions. This is what makes IDs stable and useful as handles in tuning and inspection APIs.

---

## Threading

Steps should be single-threaded internally. A step's `fit()` and `transform()` must not spawn threads, processes, or async tasks of their own. Parallelism across steps or across folds is the responsibility of the caller — not the step. Steps that internally call a library which itself parallelises (e.g. a BLAS routine) are fine, but steps must not explicitly manage their own concurrency.
