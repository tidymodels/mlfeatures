# Contributing to mlfeatures

This page covers setting up, the checks CI runs, and building the docs. Docstring, error-message, and example conventions are written up in [`CLAUDE.md`](https://github.com/tidymodels/mlfeatures/blob/main/CLAUDE.md).

## Setup

Tooling is managed with [uv](https://docs.astral.sh/uv/). The first `uv run` on a fresh clone creates `.venv` and installs the package and dev tooling from `uv.lock`, or install it up front:

```bash
uv sync
```

## Checks

Run these before opening a pull request. They mirror the CI workflow in [`.github/workflows/ci.yml`](https://github.com/tidymodels/mlfeatures/blob/main/.github/workflows/ci.yml):

```bash
uv run pytest --cov             # tests + coverage
uv run ruff check .             # lint
uv run ruff format --check .    # formatting (drop --check to apply)
uv run mypy mlfeatures extras   # type check (strict)
uv run pyright                  # type check (strict)
```

Both type checkers run in strict mode over `mlfeatures/` and `extras/`, so new code must be fully typed. CI also lints and type-checks on Python 3.10, the oldest supported version.

Install the pre-commit hooks with `uv run pre-commit install`, or run them all at once with `uv run pre-commit run --all-files`.

## Dependencies

Change dependencies with `uv add` and `uv remove`, or edit `pyproject.toml` and run `uv lock`. Commit `pyproject.toml` and `uv.lock` together, and never edit the lockfile by hand. CI installs from the lock, so it fails if the lock is stale.

## Adding a step

- Add a `tests/test_step_x_infrastructure.py` that subclasses `mlfeatures.testing.StepConformanceTest` and supplies `make_step()` and `training_data`. Every step must pass this contract, which checks the properties every step must satisfy. The `test` extra installs pytest for authors who run the contract outside this repo.
- Add a matching `step_*` method to `MlfNamespace` in [`mlfeatures/polars.py`](https://github.com/tidymodels/mlfeatures/blob/main/mlfeatures/polars.py) so the step is reachable through the `df.mlf` accessor. Tests fail if a public step has no wrapper.
- Update the `reference:` section of [`great-docs.yml`](https://github.com/tidymodels/mlfeatures/blob/main/great-docs.yml) when a public export is added or renamed.
- Write docstring examples as executable `{python}` cells, not `>>>` blocks. [`tests/test_doc_examples.py`](https://github.com/tidymodels/mlfeatures/blob/main/tests/test_doc_examples.py) runs every cell.

## Documentation

The docs site is built with [great-docs](https://github.com/posit-dev/great-docs), which needs Python 3.11 or later and [Quarto](https://quarto.org/docs/get-started/). The `examples` extra installs what the executable example pages import (xgboost and seaborn):

```bash
uv sync --group docs --extra examples
uv run great-docs build    # build into great-docs/
uv run great-docs preview  # serve at localhost:3000
```

`README.md` is generated from `README.qmd`. Edit the `.qmd` file, then re-render it:

```bash
uv run --group docs quarto render README.qmd
```

CI builds the site on every push and pull request. Pushes to `main` deploy it to Netlify; GitHub Pages publishing is switched off while the repo is private.
