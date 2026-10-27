# mlfeatures


<!-- README.md is generated from README.qmd. Edit that file, then run:
     uv run --group docs quarto render README.qmd -->

A framework for declaring, estimating, and applying feature-engineering
pipelines. It is a Python port of the R
[recipes](https://recipes.tidymodels.org/) package.

## Why mlfeatures

Preprocessing quietly goes wrong when statistics leak from test data,
the same cleanup lives in two places, or hard-coded column lists drift
from the data. A recipe declares it once, sequentially as a series of
operations done on a selection of features. Fits on training data, and
applies it identically everywhere.

- Train/serve parity: statistics are estimated once on training data and
  reused, so test data never leaks into the fit.
- Selectors: pick columns by name, role, or type instead of hard-coded
  lists, so steps keep working when the schema changes or an earlier
  step adds columns.
- Lineage: see which columns each step resolved to and where every
  output column came from.
- Serialization: save a fitted recipe to JSON or YAML and load it back
  for deployment.
- Your stack: pandas and polars through
  [narwhals](https://narwhals-dev.github.io/narwhals/), scikit-learn
  pipelines with `RecipeTransformer`, and a `df.mlf` accessor for
  polars.
- Built-in steps: the operations most models need, from imputing missing
  values and encoding categories to scaling, date features, and
  dimensionality reduction.

## Installation

Install mlfeatures from PyPI:

``` bash
pip install mlfeatures
```

mlfeatures requires Python 3.10 or later, and installs narwhals, pandas,
polars, NumPy, and scikit-learn with it.

A few features need an optional package. Install the matching extra, for
example `pip install "mlfeatures[scipy]"`:

| Extra      | Needed for                                                     |
|------------|----------------------------------------------------------------|
| `yaml`     | `to_yaml()` and `from_yaml()`                                  |
| `scipy`    | `StepOrderNorm`, `StepSplineNatural`, and `StepSplineMonotone` |
| `holidays` | `StepHoliday`, when flagging named holidays                    |
| `pyarrow`  | Date columns in pandas frames                                  |

## Lifecycle

`mlfeatures` separates feature engineering into three phases:

1.  Specify: declare a sequence of steps against a schema of variables.
    No data is touched.
2.  Fit: estimate any required statistics from a training dataset.
3.  Transform: apply the fitted recipe identically to train, test, or
    new data.

``` python
import pandas as pd
from mlfeatures import Recipe, StepDropCorr, StepNormalize, StepPca
from mlfeatures import Selectors as s

df = pd.DataFrame(
    {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated with `a`
        "c": [5.0, 1.0, 3.0, 1.0, 9.0],
    }
)

# specify -> fit -> transform
recipe = (
    Recipe()
    .add(StepDropCorr(s.numeric_features(), threshold=0.9))
    .add(StepNormalize(s.numeric_features()))
    .add(StepPca(s.numeric_features(), num_comp=1))
)
fitted = recipe.fit(df)
result = fitted.transform(df)
print(result)
```

            PC1
    0 -0.640881
    1 -1.038822
    2 -0.169031
    3 -0.144394
    4  1.993128

## Learn more

- [Introduction](https://tidymodels.github.io/mlfeatures/user-guide/getting-started.html):
  a quickstart.
- [Recipe: specify, fit,
  transform](https://tidymodels.github.io/mlfeatures/user-guide/basics/recipe-specify-fit-transform.html):
  the three phases in detail.
- [Selectors and
  roles](https://tidymodels.github.io/mlfeatures/user-guide/basics/selectors-and-roles.html):
  how steps choose their columns.
- [Arguments shared across
  steps](https://tidymodels.github.io/mlfeatures/user-guide/common-step-arguments.html):
  `columns`, `keep_original_cols`, `role`, and `id`.
- [With
  polars](https://tidymodels.github.io/mlfeatures/user-guide/polars-extension.html)
  and [with scikit-learn
  pipelines](https://tidymodels.github.io/mlfeatures/user-guide/sklearn-pipelines.html):
  using recipes in the rest of your stack.
- [Examples](https://tidymodels.github.io/mlfeatures/examples/index.html):
  six complete workflows, each with an R and Python side-by-side.
- [Vocabulary](https://tidymodels.github.io/mlfeatures/user-guide/vocabulary.html):
  R recipes and scikit-learn terms, side by side.
- [Reference](https://tidymodels.github.io/mlfeatures/reference/index.html):
  every step and class.
- [Design](https://github.com/tidymodels/mlfeatures/blob/main/design/README.md):
  the full specification.

## Development

``` bash
uv sync   # install the package + dev tooling (the dev group syncs by default)

uv run ruff check .             # lint
uv run ruff format --check .    # formatting
uv run mypy mlfeatures extras   # type check (strict)
uv run pyright                  # type check (strict)
uv run pytest --cov             # tests + coverage
```

Install the pre-commit hooks with `uv run pre-commit install`, or run
them all at once with `uv run pre-commit run --all-files`.

## Documentation

The docs site is built with
[great-docs](https://github.com/posit-dev/great-docs), which needs
Python \>= 3.11 and [Quarto](https://quarto.org/docs/get-started/):

``` bash
uv sync --group docs       # install great-docs
uv run great-docs build    # build into great-docs/
uv run great-docs preview  # serve at localhost:3000
```

CI builds the site on every push and pull request. Pushes to `main`
deploy it to Netlify; GitHub Pages publishing is switched off while the
repo is private.

## License

MIT
