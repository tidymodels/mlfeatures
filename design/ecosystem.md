# Ecosystem Fit

→ See also: [serialization.md](serialization.md), [schema.md](schema.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


## Dataframe support

The primary design constraint is **scikit-learn compatibility**: the output of `transform()` must be consumable by any sklearn estimator. That means returning a 2D numeric array or a type sklearn accepts at its boundary (pandas DataFrame, numpy array).

Beyond that boundary, users may pass in whatever dataframe type they have. Rather than writing separate code paths for polars, pandas, and future libraries, we should evaluate **[narwhals](https://github.com/narwhals-dev/narwhals)** as the internal abstraction layer. Narwhals provides a dataframe-agnostic API that works across polars, pandas, cuDF, Modin, and others, allowing steps to be written once against a stable interface.

The decision between narwhals as full internal representation vs. a thin at-boundary adapter is an **open question** to resolve before v0.1. Key considerations:

- If narwhals covers enough of what steps need (column selection, type-aware operations, categorical handling), it removes the need to commit to polars internally.
- If narwhals proves too thin, steps may still need a concrete internal frame type — polars is the natural candidate given its performance and type richness.
- Either way, the sklearn boundary contract is fixed: `transform()` returns something sklearn can consume.

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

Sparse-ness is a **column attribute** that flows through the column-lineage DAG (see [schema.md](schema.md)). It is not a mode that is converted internally — that approach was benchmarked and self-closed in R [#515].

Each step declares one of:
- `produces_sparse = True` — output is sparse (e.g. `step_dummy` on high-cardinality columns).
- `preserves_sparse = True` — passes sparse columns through unchanged.
- `requires_dense = True` — densifies input; the recipe **warns or errors** when this would destroy a user's sparse column.

`step_dummy` and any high-cardinality output are sparse by default for wide inputs. The `sparse=` argument takes `"auto" | "yes" | "no"` (not a boolean).

Sparse output format: `scipy.sparse`, mirroring `sparsevctrs` in R.

---

## Sklearn adapter

A prepped `Recipe` exposes `fit`, `transform`, `fit_transform` so it can drop into `sklearn.pipeline.Pipeline` as a single transformer:

```python
from pyrecipes.compat import RecipeTransformer

pipe = Pipeline([
    ("recipe", RecipeTransformer(rec)),
    ("model",  LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

This is the bridge to the existing Python ML ecosystem. The adapter wraps the `fit` / `transform` lifecycle; it does not re-implement it.
