# shoez

A framework for declaring, estimating, and applying feature-engineering pipelines — a Python port of the R [recipes](https://recipes.tidymodels.org/) package.

> **Status:** Pre-alpha. The package is under active design; see [`design/`](design/) for the specification.

## Lifecycle

`shoez` separates feature engineering into three phases:

1. **Specify** — declare a sequence of steps against a schema of variables. No data is touched.
2. **Fit** — estimate any required statistics from a training dataset.
3. **Transform** — apply the fitted recipe identically to train, test, or new data.

See [`design/README.md`](design/README.md) for the full design.

## Development

```bash
uv venv
uv pip install -e ".[dev]"

ruff check .          # lint
ruff format --check . # formatting
mypy shoez            # type check (strict)
pyright               # type check (strict)
pytest --cov          # tests + coverage
```

Install the pre-commit hooks with `pre-commit install`.

## License

MIT
