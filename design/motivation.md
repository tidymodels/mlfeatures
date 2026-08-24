# Motivation: Why Not Just a Pipeline?

→ See also: [selectors.md](selectors.md), [roles.md](roles.md), [schema.md](schema.md), [ecosystem.md](ecosystem.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Where pipelines get awkward: **mixed feature types needing different handling, plus
a later step operating on columns an earlier step produced.**

## Original Conception

There are a few motivations for recipes that were proposed in September of 2016:

- Enable the option to include all preprocessing steps (even deterministic ones) in one object. This object can be reused and extended.
- Enable dynamic determinations of which columns in each preprocessing step.

The first motivator is meant to enhance reproducibility and traceability rather than efficiency. We've experienced cases where different operations are performed on data in different places, leading to potential deployment issues.

Different models have different preprocessing requirements. For this reason, we can add more steps to an existing recipe to reuse existing work.

If reproducibility is an important feature, recipes should include steps for most of the operations people would perform on the feature data before modeling.

The second motivator is much more important as it enables proper methodology and fault tolerance.

Suppose we have categorical feature `A` and six other numeric features `B` through `G`. We envision a pipeline of steps:

1. Generate binary indicators for column `A`
2. Pool infrequently occurring indicators from `A` into a new "other" indicator.
3. Create interaction terms for all `A`-specific indicators and feature `B`.
4. Remove any features with a single unique value.
5. Center and scale all numeric features.
6. Feature extract features `C` through `G` to three PCA components.

Potential problems:

 - How do we know the names of the indicators in step 3 to make interactions?
 - How do we know which features pass the filters in steps 2 and 4?

The answer is that we don't unless we've 1) already run the preprocess and 2) have the same data.

We might execute this preprocessing with different data, perhaps during cross-validation. If we've hard-coded specific column names, the preprocessor could fail after steps 2 and 4 since there could be variation in which features are retained (or, in other steps, created). We could write these steps more generally as:

1. Generate binary indicators for _all categorical features_.
2. Pool infrequently occurring _all features starting with `A`_ into a new "other" indicator.
3. Create interaction terms for all _any features starting with `A_` _ and feature `B`.
4. Remove _all features_ with a single unique value.
5. Center and scale _any numeric features_.
6. Feature extract _any features with names `C` through `G`_ to three PCA components.

The system is dynamic in the sense that it can avoid hardcoding which variables are affected or created by specific processing steps. This was intended to emulate the "garden of forking paths," where, depending on certain decisions or the current data, the processing can look very different downstream.

## The scenario

A housing dataset with mixed column types:

| column | type | needs |
| ------------------------------------ | ----------- | ----------------------------------------- |
| `price` | float | the target |
| `sqft`, `lot_size`, `year_built` | numeric | impute → normalize |
| `geo_lat`, `geo_lon`, `geo_elevation` | numeric | impute → **PCA with the dummies** |
| `neighborhood`, `style`, `roof` | categorical | impute `"unknown"` → one-hot encode |

The twist: PCA the dummies **together with just the `geo_*` numerics** (not `sqft` /
`lot_size` / `year_built`), then normalize **everything numeric** — including the new
PCA components.

## The sklearn way

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

numeric_plain = ["sqft", "lot_size", "year_built"]
numeric_geo   = ["geo_lat", "geo_lon", "geo_elevation"]
categorical   = ["neighborhood", "style", "roof"]

plain_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

pca_branch = Pipeline([
    ("split", ColumnTransformer([
        ("geo", SimpleImputer(strategy="median"), numeric_geo),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("dummy", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])),
    ("pca", PCA(n_components=10)),
])

pre = ColumnTransformer([
    ("plain", plain_pipe, numeric_plain),
    ("pca",   pca_branch, numeric_geo + categorical),
])

pipe = Pipeline([("pre", pre), ("model", ...)])
```

- PCA over a numeric subset + the dummies forces a **nested** `ColumnTransformer` and
  splits the numerics into two hand-maintained lists.
- Normalizing the PCA output *and* the plain numerics together isn't possible — the
  branches run in parallel; you'd nest yet another pipeline with a third list.
- Three literal column lists, drifting from reality on every schema change. No "all
  numeric features", no "everything starting with `geo_`".
- Output is a bare array — provenance and a train/serve column contract are gone.

## The shoez way

```python
rec = (
    Recipe(data=df_train, target=["price"])
    .add(ImputeMedian(s.numeric_features()))
    .add(Unknown(s.categorical_features()))
    .add(Dummy(s.categorical_features()))
    .add(PCA(
        has_source(step_type=step_dummy) | starts_with("geo_"),
        num_comp=10,
    ))
    .add(Normalize(s.numeric_features()))
)

trained = rec.fit(df_train)
X_train = trained.transform(df_train)
X_test  = trained.transform(df_test)
```

- Selectors are a **live query over a schema that updates after every step** — the
  final `s.numeric_features()` automatically includes the PCA components.
- `has_source(step_type=step_dummy) | starts_with("geo_")` is one declarative union:
  the dummies *plus* the `geo_*` numerics, no list to partition. `sqft` / `lot_size` /
  `year_built` match neither and flow past PCA untouched.
- Linear top-to-bottom order; no nesting, no parallel branches to reconcile.
- Provenance survives — `repr(rec)` shows `step_dummy [roof → roof_tile, ...]`.

## Selectors vs. hardcoded names

The single biggest source of quiet breakage in a preprocessing pipeline is a literal
list of column names. A hardcoded list is a claim about the data that nothing checks
and nothing updates, and it is wrong in four ways that a selector is not.

**It is a copy of the schema that drifts.** `numeric_plain = ["sqft", "lot_size",
"year_built"]` is correct until someone adds `garage_sqft` to the source table. Then
the new column is silently not imputed and not normalized: no error, no warning, just
a model fit on a column the pipeline never touched. `s.numeric_features()` is a
question asked against the schema at fit time, so it has no stale copy to drift from.

**It cannot name columns that do not exist yet.** Steps create columns. One-hot
encoding turns `roof` into however many indicators the training split happens to
contain; PCA emits `PC1 … PC10`. Writing the next step against literal names means
knowing the output of the previous step, which means having already run it on this
exact data. Under cross-validation or a fresh data pull that assumption fails, and it
fails per-resample: a rare `roof` level present in fold 1 and absent in fold 2 gives
you a pipeline that works four times out of five. A selector is evaluated after each
step against the schema as it then stands, so `s.numeric_features()` in the final
`Normalize` picks up the PCA components without anyone writing `PC1` anywhere.

**It cannot express "whatever survived".** Filter steps are defined by the columns
they remove: drop zero-variance columns, drop one of each correlated pair. Which
columns those are is a property of the data, not of the code, and it differs between
resamples by construction. There is no literal list that is correct for every split;
there is only the query "the features still numeric at this point".

**It says how, not why.** `["geo_lat", "geo_lon", "geo_elevation"]` and
`starts_with("geo_")` select the same three columns today, but only the second records
the intent. A reader (and a reviewer) can see that the step is about geographic
features, and a fourth `geo_*` column joins the group automatically instead of
requiring a search for every list that ought to have mentioned it.

The cost is real and worth stating: a selector defers to fit time what a list settles
at authoring time, so a typo'd or over-broad selector surfaces later than a typo'd
name would, and reading the code no longer tells you exactly which columns a step
touched. That is what the schema and the fitted recipe's `repr` are for: after
`fit()`, every step reports the concrete columns it resolved to. The selector
describes the rule; the fitted recipe records the answer.
