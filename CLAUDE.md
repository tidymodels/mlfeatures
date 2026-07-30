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

When adding a new step, also add a matching `step_*` wrapper method to the `ShzNamespace` in [`shoez/polars.py`](shoez/polars.py) so the step is reachable through the `df.shz` accessor. Mirror the step's constructor signature and keep the one-line docstring pointing at the step class (e.g. `See \`StepDummy\`.`); without a wrapper the step is only reachable via the generic `df.shz.apply(...)`. The `TestWrapperStepParity` tests in [`tests/test_polars_accessor.py`](tests/test_polars_accessor.py) enforce this both ways: CI fails if a public step has no wrapper, or if a wrapper points at a step that no longer exists, so the accessor can never silently drift out of sync.

## Error and warning messages

Raised exceptions and `warnings.warn` messages follow a uniform style so they read consistently and match Python's own conventions:

- **Lowercase first word**, unless it is an identifier that is naturally capitalized (e.g. a class name like `StepDummy` in the `called before fit()` guards).
- **No trailing period.** Join what would be two sentences with `;` rather than `. ` so the message stays a single phrase.
- **Single quotes** around parameter names, role names, and values (`'degree'`, `'outcome'`, `got 'kendall'`) — use `{value!r}` for interpolated values. Do not use backticks (a Markdown convention, not a runtime one).
- **Numerals** for counts (`fewer than 2 unique levels`, not `two`).
- **No em dashes** — use `;` or `:` instead.
- Pluralize with the `(s)` form (`column(s)`, `name(s)`) and quote column lists as their repr (`['empty']`).

Exception: the sklearn-compat layer (`compat.py`) raises `NotFittedError` with sklearn's own house style (capitalized, trailing period) so those messages read like the rest of the sklearn ecosystem; only the quoting rule applies there.

## Documentation

The documentation site is built with [`great-docs`](https://github.com/posit-dev/great-docs), configured in [`great-docs.yml`](great-docs.yml). It auto-discovers the public API from `shoez.__all__`; when you add or rename a public export, update the `reference:` section in `great-docs.yml` to match.

```bash
uv run great-docs build      # build the site into great-docs/ (gitignored)
uv run great-docs preview    # build and serve at localhost:3000
```

`great-docs` is in the `docs` dependency group and requires Python ≥3.11 (the package itself still supports ≥3.10). On push to `main`, [`.github/workflows/docs.yml`](.github/workflows/docs.yml) builds and deploys to GitHub Pages.

Write docstring `Examples` as executable ```` ```{python} ```` cells, never `>>>` doctest blocks (great-docs runs the cells and embeds their real output; `>>>` blocks render verbatim and drift silently). See `shoez/schema.py` for the pattern. Those cells are executed on every run of the test suite by [`tests/test_doc_examples.py`](tests/test_doc_examples.py), which discovers each `{python}` cell and asserts it runs without raising, so a broken example fails a normal PR check rather than only the `main` docs deploy.

### Example style

These conventions apply to every example we write: docstring `Examples` cells, [`README.qmd`](README.qmd), and the snippets in `design/`.

- **Import selectors as `s`**: `from shoez import Selectors as s`, then `s.numeric_predictors()`. This is the house style everywhere, and it keeps the shoez namespace clear of `cs`, the conventional alias for `polars.selectors`. Prose cross-references to the API still use the real name (`Selectors.wrap`).
- **Select columns with selectors, not literal names.** Pass `s.numeric_predictors()` rather than a hardcoded list like `["a", "b", "c"]`; the selector says what the step is for, and the example keeps working when the frame changes.
- **Build recipes by chaining `.add()`** rather than `Recipe(steps=[...])`, binding the recipe to a name and fitting on the next line. A single step stays on one line (`recipe = Recipe().add(StepNormalize(s.numeric_predictors()))`); a step's own docstring needs only that step. In overview examples (the README, `Recipe`) show at least two steps, one per line inside a parenthesized expression, so the chained shape is the thing being demonstrated:

  ```python
  recipe = (
      Recipe()
      .add(StepDropCorr(s.numeric_predictors(), threshold=0.9))
      .add(StepNormalize(s.numeric_predictors()))
  )
  ```

- **Keep the three lifecycle phases on their own lines.** Bind each one rather than chaining them together, so specify, fit, and transform read as the distinct phases they are:

  ```python
  recipe = Recipe().add(StepNormalize(s.numeric_predictors()))
  fitted = recipe.fit(df)
  fitted.transform(df)
  ```

- **End on the real result.** The final expression should display the actual output (a bare `result`), not `.shape`, `.agg(...)`, or another summary standing in for it, and it needs no trailing comment explaining what the reader is about to see.
