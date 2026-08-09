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

### Vocabulary

Describe a recipe's two states as **untrained** and **trained**, never as "prepped"/"unprepped" or "prepared". Those are inherited from R `recipes`' cooking analogy, which shoez does not use; the repr headers (`Recipe (untrained)` / `Recipe (trained)`) and the per-step `[untrained]` / `[trained]` markers are the wording to match. For the same reason, call the second lifecycle phase `transform()` rather than "bake". The one exception is when citing R `recipes`' actual function names (its `prep()` and `bake()`), as `shoez/step_normalize.py` does.

### Step docstrings

A step's class docstring is its reference page, so it is written for someone using the step, not for someone maintaining it. Module docstrings and the `fit_*`/`apply_*` helper docstrings are read from the source instead, so they may talk about internals freely.

- **Only cross-reference names in `shoez.__all__`.** great-docs makes pages from the public API, so a single-backtick reference to anything else (the unexported `fit_*`/`apply_*` helpers, a type alias like `Metric`) renders as inert code text pointing nowhere. Spell the values out instead: "Distance function: ``"euclidean"`` (default) or ``"manhattan"``".
- **No framework vocabulary.** Don't name `Mode` members (`ONE_TO_ONE`, `ONE_TO_MANY`, `MANY_TO_MANY`, `REMOVING`, `ROW_OP`, `ADDING`, `UNCONSTRAINED`) or base classes like `RowOp` in a step's docstring; describe what the step does to the data. This is a rule about step docstrings only: `Mode` is a public export with its own reference page, and the Mode-keyed tables in `design/` are how the design docs are meant to read.
- **Universal contracts live once on `Step`, and steps don't restate them.** Three currently do: everything a step needs is computed at fit time and `transform()` only applies stored state; a `columns` selector is resolved once at fit time, and a selection matching nothing makes the step the identity transform; a step guarantees *which* columns exist and what they are named, not where they sit. A new invariant that holds for every step belongs there too.
- **Don't promise column positions.** Say what columns a step adds or removes, never that they are "appended at the end" or replaced "in the same positional slot".
- **Give every step a `Parameters` section** listing exactly its constructor parameters, in signature order, each with its default.
- **Describe `columns` with the shared wording**, varying only the verb, the example selector, and the dtype sentence:

  ```
  columns
      Columns to project. Accepts a single column name, a list of names, a
      shoez selector such as ``Selectors.numeric_predictors()`` (see
      `Selectors` for the full set), or a narwhals or polars selector
      (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
      Selectors are resolved against the frame's schema when the recipe is
      fitted, so the same selector works on any backend. Only numeric columns
      are meaningful here.
  ```

- **Say which dtypes are meaningful, not that others are rejected.** `accepts_dtypes()` is declared per step but nothing calls it, so a wrong dtype surfaces as a backend error rather than a shoez one. Until that is wired up, docstrings must not claim validation.
- **Leave out what the reader can already see.** A filter not touching columns, or a constant column having nothing to estimate, does not need a paragraph.

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

- **One `{python}` cell per output.** A cell auto-displays only its last expression, so a value you want rendered (the input frame, the untrained recipe, the fitted recipe) needs its own cell with a line of prose between. Don't reach for `print()` or `display()` to force intermediate output; splitting the cell keeps the rich rendering and the copy-pasteable code. [`README.qmd`](README.qmd) is the exception: it renders to GitHub-flavored Markdown, where a DataFrame's `_repr_html_` drags a raw `<style>` block into `README.md`, so `print()` the frame there.
- **Show arguments that change behavior.** After the walkthrough, add a cell per argument whose effect is visible in the output (`keep_original_cols`, `num_comp`, `threshold`, `degree`), each with at most one line of prose. Arguments that only rename or re-role columns don't need one.
- **End on the real result.** The final expression should display the actual output (a bare `result`), not `.shape`, `.agg(...)`, or another summary standing in for it, and it needs no trailing comment explaining what the reader is about to see.
