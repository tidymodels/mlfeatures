# mlfeatures


<!-- README.md is generated from README.qmd. Edit that file, then run:
     uv run --group docs quarto render README.qmd -->

A framework for declaring, estimating, and applying feature-engineering
pipelines. It is a Python port of the R
[recipes](https://recipes.tidymodels.org/) package.

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

See
[`design/README.md`](https://github.com/tidymodels/mlfeatures/blob/main/design/README.md)
for the full design.

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
