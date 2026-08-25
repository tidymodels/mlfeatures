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

### Declaring per-output provenance: `output_parents()`

The mode is a coarse shape; it does not say *which* input produced *which* output, and that distinction is real and mode-dependent. Each `step_dummy` indicator descends from a single source column (`a1, a2 ← a`), whereas every `step_pca` component descends from the whole input set (`pc1, pc2 ← {a, b, c}`), and an `ADDING` step like `step_intercept` has no parents at all. So every step that *creates* columns declares the exact mapping:

```python
def output_parents(self) -> dict[str, list[str]]:
    """Map each produced column to the input column(s) it descends from."""
```

Known after `fit()`. The keys are exactly the columns the step adds to the frame; the recipe validates them against the post-step frame and rejects a mismatch. The base-class default returns `{}` — correct for steps that create no new columns (in-place `ONE_TO_ONE`, `REMOVING`, `ROW_OP`). A degree-0 `step_poly` column is a pass-through, not a newly produced column, so it is excluded from the map.

### Declaring added columns for collision checking: `produced_columns()`

`output_parents()` keys are necessarily unique (it is a `dict`), so they cannot express a step that would emit the *same* name twice. `produced_columns()` fills that gap: an ordered list of the names a step adds, **preserving duplicates**.

```python
def produced_columns(self) -> list[str]:
    """Ordered names this step adds, with duplicates preserved."""
```

The recipe passes this list to the single centralized name-collision check (`Schema.check_additions`) **before** the step runs, rejecting both a name that already exists in the schema and a name the step would emit more than once. Validating up front keeps the error clean and identical across backends: at transform time polars raises a backend-specific error while pandas silently keeps only one of the colliding columns, so a post-transform check cannot see a within-step duplicate. The base-class default derives the list from `output_parents()` keys; steps with a free-form naming scheme that could self-collide (`step_dummy` / `step_one_hot`) override it to preserve duplicates. Steps no longer perform their own collision checks.

### What recording this information enables

After `fit()`, the lineage DAG contains the full input/output mapping for every step. This unlocks:

- **Feature provenance.** The lineage DAG records which input columns produced which output columns. Tracing PCA components or dummy columns back to their source is possible programmatically. This is the core problem identified in [tidymodels/recipes#1158](https://github.com/tidymodels/recipes/issues/1158).
- **`has_source()` selector.** Users can select all columns produced by a given step or step type — e.g. `has_source(step_type=step_dummy)`. See [selectors.md](selectors.md).
- **Naming contract enforcement.** The framework validates that generated column names are unique — both against the existing schema and within the step — before each step runs, via `produced_columns()` and `Schema.check_additions`. Steps do not self-check.
- **`keep_original_cols` applicability.** Every `ONE_TO_MANY` and `MANY_TO_MANY` step exposes `keep_original_cols` as a constructor argument defaulting to `False`, since keeping the source columns beside the derived ones is a legitimate thing to want. Steps in the other modes fix the value in `__init__`: `ONE_TO_ONE` and `REMOVING` steps to `False`, `ADDING`, `ROW_OP`, and `UNCONSTRAINED` steps to `True`.

---

## `fit()`

- Resolves selectors against the current schema.
- Estimates and stores statistics. By convention a step holds its fitted statistics on a single `_estimates` attribute, initialized to `None` in `__init__` and populated in `fit()`; steps with more than one piece of fitted state bundle it into one frozen dataclass (e.g. `DummyEstimates`, `PolynomialFit`). This is a convention, not a base-class slot — the stored type varies per step, so `Step` declares no `_estimates`.
- Returns the fitted step. Setting `trained=True` is done by `recipe.fit()` after calling `step.fit()` — not by the step itself.
- Declares `schema_in()` and `schema_out()` so the recipe can update its column-lineage DAG. See [schema.md](schema.md).
- Must be idempotent given the same training data.

---

## `transform()`

- Applies stored statistics to `new_data`. Never re-estimates.
- Wrapped by the base class to add exception chaining: `raise StepError(step_id=self.id, step_type=type(self).__name__, columns=...) from underlying_exc`. See [inspection.md](inspection.md).
- Sparse-in → sparse-out (or explicit densification warning) where declared. See [ecosystem.md](ecosystem.md).

---

## Empty selection ⇒ identity (structural)

An **empty column selection is a no-op**: a step whose primary selector resolves to zero columns passes the frame through unchanged. Like the `RowOp` transform-time skip above, this is enforced **structurally by the framework, not by a per-step convention** — a step author cannot get it wrong because the step is never fitted on an empty selection.

The mechanism, all at the recipe boundary:

- `Step.resolve_selection(data, info) -> list[str]` resolves the step's *primary input selector* to concrete column names. The base default resolves `self.columns`, which is correct for every selector-taking step — including multi-selector steps like `step_class_distance`, whose primary (feature) selector is `columns` (the `target` selector does not gate the no-op). It takes `data` (not just `info`) because narwhals / polars selectors resolve against the frame's schema.
- `recipe.fit()` calls `resolve_selection` *before* `fit()`. If the result is `[]`, it sets `Step._empty_selection = True`, sets `trained`, keeps the step in the fitted pipeline (so it stays inspectable), and **skips** `fit()` / `transform()` / `update_roles()`. The marker is reset each fit, so re-fitting with a selection that now matches columns is not stuck in identity mode.
- `FittedRecipe.transform()` skips any step with `_empty_selection` set, alongside `RowOp` steps.

Steps with **no input selector** (e.g. `step_intercept`, an `ADDING` step with no `columns`) must always run; the recipe special-cases them so `resolve_selection` is never called on them and the marker is never set.

Because the no-op lives in the recipe, individual steps' `fit()` code must **not** add their own empty-selection guards — `fit()` is guaranteed a non-empty selection.

---

## `StepConformanceTest`

Implemented as `shoez.testing.StepConformanceTest`. It ships **inside the package**, not under `tests/`, so a third-party step can be held to the same contract as a first-party one. Subclass it, supply `make_step()` and `training_data`, and pytest collects the whole contract:

```python
from shoez.testing import StepConformanceTest

class TestMyStepConformance(StepConformanceTest):
    training_data = {"a": [1.0, 2.0, 3.0], "g": ["x", "y", "x"]}

    def make_step(self) -> StepMyThing:
        return StepMyThing(["a"])
```

Every step must pass it before merging:

- Empty selection returns empty output without error (handled structurally by the recipe — see [Empty selection ⇒ identity](#empty-selection--identity-structural); the step's own `fit()` is never called on an empty selection).
- 0-row and 1-row `transform()` produce schema-correct output.
- `fit()` is idempotent given the same training data.
- `transform()` never re-estimates: transforming a subset of the training frame equals transforming the whole frame and then taking the subset. Every other check runs against the training frame itself, where a step that recomputes its statistics is invisible because the recomputed values agree; this is the check that sees it.
- Sparse-in → sparse-out (or explicit densification warning) where declared.
- Pickle round-trip preserves `transform()` output.
- JSON round-trip preserves `transform()` output, or — for a step carrying user callables it needs at transform time, like `step_mutate` — is refused loudly. Set `supports_json = False` to assert the refusal.
- `id` is stable across sessions for the same hyperparameters.
- `output_parents()` declares exactly the columns the step adds, each with parents drawn from the input frame.
- Backend parity: fitting and transforming the same data on a pandas frame and a polars frame produces identical narwhals output — same column names, same dtypes, and the same values (missing-aware, floats compared approximately). This guards against backend-divergence bugs like the `lit(None)` → `Object` footgun PR #74 fixed for `StepDummy`, where pandas emitted object-dtype indicators while polars stayed numeric, and against a step computing different numbers depending on the backend it fitted on.

A check that genuinely does not apply is opted out of through a hook, never by omission, so "considered and not applicable" stays distinguishable from "forgotten": `make_empty_step()` returning `None` for a step with no column selector, and `estimates_attr = None` for a step that estimates nothing. Both then skip with an explicit reason.

Not yet covered: constructor arguments are not type-validated by the suite.

---

## Step IDs

Every step has a stable `id`, defaulting to the step type's short name (`"normalize"`) and user-overridable via the `id` constructor argument. All inspection and tuning APIs key on `id`, never on position. Position-based indexing is deprecated from the start.

IDs must **not** use randomness: the same recipe, written the same way, must produce the same ids across sessions. That is what makes them usable as handles in tuning and inspection APIs, and it is what lets a `Selectors.has_source("dummy")` be written by hand against a recipe you can read.

Ids are unique within a recipe, enforced by `Recipe.add` (and by the `Recipe(steps=[...])` constructor, which is the same thing) at the earliest point a collision is detectable. Uniqueness matters because the id is the key for the lineage DAG's `source` field, `SchemaInfo.producing_steps`, the `has_source` / `has_ancestor` selectors, and step-level error messages; two steps sharing an id turn each of those into a wrong answer rather than an error.

Two steps of the same type are resolved differently depending on where the id came from:

| Id | On collision |
| --- | --- |
| Defaulted (the type's short name) | Silently suffixed with the lowest free `_n`: `"dummy"`, then `"dummy_2"`. The renaming is done on a copy, so `add()` stays pure with respect to the caller's step objects. |
| Passed explicitly | `ValueError`. The caller named the step in order to refer to it, so a duplicate is a mistake about that name, not something to paper over. |

`Step._init_id()` records which of the two a step's id is; concrete steps call it rather than assigning `self.id` directly.

---

## Threading

Steps should be single-threaded internally. A step's `fit()` and `transform()` must not spawn threads, processes, or async tasks of their own. Parallelism across steps or across folds is the responsibility of the caller — not the step. Steps that internally call a library which itself parallelises (e.g. a BLAS routine) are fine, but steps must not explicitly manage their own concurrency.
