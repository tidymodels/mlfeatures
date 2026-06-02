# shoez

A framework for declaring, estimating, and applying feature-engineering pipelines.

## Design

The full design specification lives in [`design/README.md`](design/README.md). Read it before working on the package — it documents the lifecycle (specify/fit/transform), core concepts (roles, types, selectors, steps), and links to detailed docs for each subsystem.

## Development

Tooling is managed with [`uv`](https://docs.astral.sh/uv/); always invoke tools via `uv run` so they use the project venv. Run these checks after writing or changing code — they mirror the CI pipeline in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), so passing them locally means CI should pass:

```bash
uv run pytest                  # tests (add --cov --cov-report=term-missing for coverage)
uv run ruff check .            # lint
uv run ruff format --check .   # format check (drop --check to apply)
uv run mypy shoez              # type-check (strict)
uv run pyright                 # type-check (strict)
```

Both type-checkers run in strict mode over `shoez/` (tests are excluded). New code must be fully typed and pass both. A [`.pre-commit-config.yaml`](.pre-commit-config.yaml) wires ruff and mypy into git hooks; `uv run pre-commit run --all-files` runs them all at once.

When adding a new step, also run its `StepConformanceTest` (see [`design/step-contract.md`](design/step-contract.md)) — every step must pass it before merging.

## Documentation

The documentation site is built with [`great-docs`](https://github.com/posit-dev/great-docs), configured in [`great-docs.yml`](great-docs.yml). It auto-discovers the public API from `shoez.__all__`; when you add or rename a public export, update the `reference:` section in `great-docs.yml` to match.

```bash
uv run great-docs build      # build the site into great-docs/ (gitignored)
uv run great-docs preview    # build and serve at localhost:3000
```

`great-docs` is in the `docs` dependency group and requires Python ≥3.11 (the package itself still supports ≥3.10). On push to `main`, [`.github/workflows/docs.yml`](.github/workflows/docs.yml) builds and deploys to GitHub Pages.
