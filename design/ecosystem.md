# Ecosystem Fit

→ See also: [serialization.md](serialization.md), [lineage.md](lineage.md), [frame-notes.md](frame-notes.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Dataframe support

The primary design constraint is **scikit-learn compatibility**: the output of `transform()` must be consumable by any sklearn estimator. That means returning a 2D numeric array or a type sklearn accepts at its boundary (pandas DataFrame, numpy array).

Beyond that boundary, users may pass in whatever dataframe type they have. Rather than writing separate code paths for polars, pandas, and future libraries, we use **[narwhals](https://github.com/narwhals-dev/narwhals)** as the internal abstraction layer. Narwhals provides a dataframe-agnostic API that works across polars, pandas, cuDF, Modin, and others, allowing steps to be written once against a stable interface.

**Decision (v0.1): narwhals is the full internal representation.** Steps are written once against the narwhals API; narwhals dispatches to the user's native backend (pandas in → pandas out, polars in → polars out). We do **not** commit to polars internally, and steps do **not** carry per-backend code paths. Consequences:

- A step that needs an operation narwhals does not expose may drop to the native data frame via `nw.to_native(...)`, but every such escape is treated as a defect to upstream into narwhals, not a license to branch on backend. The prototype step set [#18] is the conformance workload that validates narwhals is thick enough — categoricals (`step_unknown`), row ops (`step_filter`), and sparse output (`step_dummy`) are the riskiest probes.
- The sklearn boundary contract is fixed regardless: `transform()` returns something sklearn can consume (numpy array or pandas DataFrame), converting out of the native type at that one boundary.
- If narwhals proves too thin during #18, the fallback is polars as a concrete internal data frame type — but that reopens this decision explicitly rather than being assumed.

Additional requirements:

- **Categorical type:** must round-trip `pandas.Categorical` (ordered + unordered) and `polars.Categorical` / `Enum`.
- **Missing values:** support both NumPy `NaN` and pandas/Arrow nullable types. Steps that fit on missing values must declare their NA policy.

### Performance

Each step is responsible for implementing its computation as efficiently as possible. There is no prescribed internal approach — steps should use whatever vectorized or batch operations the underlying library supports best.

---

## Strings stay strings

No implicit `strings_as_factors` conversion. Categorical encoding is opt-in per step. R's `strings_as_factors = TRUE` default on `fit()` (not the recipe) was its canonical "wrong default" and maintainers explicitly want to flip it [#836, #715, #1479]. We start correctly.

---

## Sparse support

> **Note:** This section is provisional. Sparse support in Python's ML ecosystem (scipy.sparse, pyarrow, polars, narwhals) is fragmented and the right approach here will require more research before anything is finalised.

Sparse-ness is a **column attribute** that flows through the column-lineage DAG (see [lineage.md](lineage.md)). It is not a mode that is converted internally — that approach was benchmarked and self-closed in R [#515].

Each step declares one of:
- `produces_sparse = True` — output is sparse (e.g. `step_dummy` on high-cardinality columns).
- `preserves_sparse = True` — passes sparse columns through unchanged.
- `requires_dense = True` — densifies input; the recipe **warns or errors** when this would destroy a user's sparse column.

`step_dummy` and any high-cardinality output are sparse by default for wide inputs. The `sparse=` argument takes `"auto" | "yes" | "no"` (not a boolean).

Sparse output format: `scipy.sparse`, mirroring `sparsevctrs` in R.

---

## Sklearn adapter

`RecipeTransformer` wraps an untrained `Recipe` and exposes `fit`, `transform`, `fit_transform` so it can drop into `sklearn.pipeline.Pipeline` as a single transformer:

```python
from mlfeatures import RecipeTransformer

pipe = Pipeline([
    ("recipe", RecipeTransformer(rec)),
    ("model",  LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

This is the bridge to the existing Python ML ecosystem. The adapter wraps the `fit` / `transform` lifecycle; it does not re-implement it.

sklearn keeps `X` and `y` apart, while a recipe identifies the target by its `"target"` role. The adapter joins `y` onto `X` under the recipe's target column name at fit time and drops the target column from `transform()` output, so it returns features only. It rejects `y` when the recipe has no target role, recipes with more than one target column, and any fitted step that reads the target column (sklearn's `transform()` never receives `y`; transform the target outside the pipeline, e.g. with `TransformedTargetRegressor`). After `fit` it sets `fitted_`, `feature_names_in_`, `n_features_in_` and caches the output names for `get_feature_names_out()` and `set_output`.

### Tuning

Step hyperparameters are exposed through sklearn's nested-parameter convention, keyed on step `id` (never position): `get_params(deep=True)` returns `recipe__<step id>__<parameter>`, and `set_params` accepts the same keys, so `GridSearchCV` can tune `recipe__corr__threshold` directly. `Recipe` and `Step` each implement `get_params` / `set_params`; both are immutable, so `set_params` returns a new object, and `RecipeTransformer.set_params` rebinds `self.recipe` to it. A step's `id` and `columns` are not tunable. `Recipe.__sklearn_clone__` makes `sklearn.base.clone` copy the recipe rather than rebuild it from its parameters.
