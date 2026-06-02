# Motivation: Why Not Just a Pipeline?

→ See also: [selectors.md](selectors.md), [roles.md](roles.md), [schema.md](schema.md), [ecosystem.md](ecosystem.md)

---

> **Note:** Code examples in this document are illustrative and not final — API names, signatures, and patterns may change before implementation.


Where pipelines get awkward: **mixed predictor types needing different handling, plus
a later step operating on columns an earlier step produced.**

## The scenario

A housing dataset with mixed column types:

| column | type | needs |
| ------------------------------------ | ----------- | ----------------------------------------- |
| `price` | float | the outcome |
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
  numeric predictors", no "everything starting with `geo_`".
- Output is a bare array — provenance and a train/serve column contract are gone.

## The pyrecipes way

```python
rec = (
    Recipe(formula="price ~ .", data=df_train)
    .step_impute_median(all_numeric_predictors())
    .step_unknown(all_categorical_predictors())
    .step_dummy(all_categorical_predictors())
    .step_pca(
        has_source(step_type=step_dummy) | starts_with("geo_"),
        num_comp=10,
    )
    .step_normalize(all_numeric_predictors())
)

prepped = rec.fit(df_train)
X_train = prepped.transform(df_train)
X_test  = prepped.transform(df_test)
```

- Selectors are a **live query over a schema that updates after every step** — the
  final `all_numeric_predictors()` automatically includes the PCA components.
- `has_source(step_type=step_dummy) | starts_with("geo_")` is one declarative union:
  the dummies *plus* the `geo_*` numerics, no list to partition. `sqft` / `lot_size` /
  `year_built` match neither and flow past PCA untouched.
- Linear top-to-bottom order; no nesting, no parallel branches to reconcile.
- Provenance survives — `repr(rec)` shows `step_dummy [roof → roof_tile, ...]`.
