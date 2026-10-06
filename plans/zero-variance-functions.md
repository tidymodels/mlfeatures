# Zero-variance functions: `fit_zero_variance` and `apply_filter`

## Context

`mlfeatures` is organised around a `Step` ABC with `fit` / `transform` methods (see
`design/step-contract.md`). The catalog (`design/steps-catalog.md`) lists
`step_zv` ("REMOVING mode: drop zero-variance columns") as the canonical
zero-variance step, alongside other REMOVING-mode steps such as `step_corr`.

This change introduces a **functional** version first: a fitter that finds
the offending columns, and a separate applier that drops them. Because
*every* REMOVING-mode step ends in "drop these columns from the frame", the
applier is shared rather than specific to zero-variance:

- `fit_zero_variance(features)` lives in `mlfeatures/step_zero_variance.py` (one
  file per step, mirroring R `recipes`' `step_zv.R` convention) and is
  specific to the zero-variance rule.
- `apply_filter(data, drop_cols)` lives in `mlfeatures/apply_shared.py` and is
  reused by every REMOVING-mode fitter (`step_zv`, `step_corr`, future
  filters).

Both functions are precursors to the eventual `step_zv` Step subclass. The
fit/apply split here mirrors the framework's fit/transform contract so the
logic can later be wrapped in a Step without rewriting it.

## Decisions

- **Dataframe library:** narwhals is the project's chosen abstraction
  (`design/ecosystem.md`: *"Decision (v0.1): narwhals is the full internal
  representation. pandas in produces pandas out, polars in produces polars
  out"*). The functions are written once against narwhals and work for any
  backend it supports. This change adds `narwhals` to the project's runtime
  dependencies; it is the first module that actually uses the abstraction.
- **Return type of `fit_zero_variance`:** a 1-D `numpy.ndarray` of strings,
  the Python analogue of an R character vector. `apply_filter` accepts any
  iterable of strings, so callers can pass this array straight through or
  use a hand-written list.
- **Scope:** strict zero-variance only (single unique value). Near-zero
  variance (R's `freq_cut` and `unique_cut` knobs) can come later.
- **NA handling:** ignore NAs. A column with one non-NA value plus NAs is
  zero-variance, matching R `recipes`' default. A column whose values are
  all NA has zero observed values and is also flagged.
- **Fit argument name:** `features` (not `data`), to signal that the input
  holds the candidate predictor columns. `apply_filter` keeps `data` since
  it is a generic column-dropping helper, not zero-variance specific.

## Files

### `mlfeatures/step_zero_variance.py`

One public function, `fit_zero_variance(features)`. Counts distinct non-null
values per column using `frame[c].drop_nulls().n_unique()` and returns the
names where the count is `<= 1`. The `<= 1` rule lets all-null columns
count as zero-variance (zero observed values, so no information either
way). The result is forced to `dtype=np.str_` so the empty-list case still
produces a string-typed array.

### `mlfeatures/apply_shared.py`

One public function, `apply_filter(data, drop_cols)`. Wraps the input via
`nw.from_native(eager_only=True)`, filters `drop_cols` to names that are
actually present in `data` (silent skip for missing names, needed for
train/serve parity when an upstream step has already dropped one of them),
calls `frame.drop(*present)`, and unwraps back via `nw.to_native`. The
`IntoFrameT` generic in the return type guarantees the backend round-trips
(pandas in produces pandas out, polars in produces polars out).

### `mlfeatures/__init__.py`

Both functions are added to the package's flat namespace and to `__all__`.

### `pyproject.toml`

Adds `narwhals>=2.0` to `[project] dependencies`. The version floor matches
the major version `uv` resolved to; the `narwhals.stable.v1` import path
the code uses is API-stable across narwhals major versions.

### `great-docs.yml`

`CLAUDE.md` instructs us to update the `reference:` section of
`great-docs.yml` when adding public exports. A new `Functions` section
lists `fit_zero_variance` and `apply_filter`.

### `tests/test_zero_variance.py` and `tests/test_apply_shared.py`

Pytest suites parametrized across pandas and polars via a `make_frame`
factory, so the narwhals layer is exercised against both backends. The
fitter suite covers: single constant column, multiple constants, varying
columns, NA handling (mixed, all-NA), single-row frames, empty frames
(pandas and polars constructed separately because polars infers dtypes
from data), string columns (anything-goes dtype), return-type checks
(numpy array, string dtype preserved when empty). The applier suite
covers: drop semantics, accepting the numpy array from the fitter, plain
list and tuple inputs, empty input as a no-op, silent skip on unknown
names, backend preservation, and a round-trip train/serve check.

## Verification

The same checks CI runs (see `CLAUDE.md`):

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy mlfeatures
uv run pyright
```

Quick manual smoke-test in a REPL:

```python
import pandas as pd, polars as pl
from mlfeatures import fit_zero_variance, apply_filter

data = {"a": [1, 1, 1], "b": [1, 2, 3], "c": ["x", "x", None]}

df_pd = pd.DataFrame(data)
fit_zero_variance(df_pd)                      # array(['a', 'c'])
apply_filter(df_pd, fit_zero_variance(df_pd)) # pandas DataFrame with just 'b'

df_pl = pl.DataFrame(data)
apply_filter(df_pl, fit_zero_variance(df_pl)) # polars.DataFrame with just 'b'
```

## Open work (tracked separately)

`fit_zero_variance` has known fault-tolerance gaps that were intentionally
deferred to a follow-up issue:

1. Crashes on pandas object columns containing unhashable values (lists,
   dicts) and on polars `Object` dtype columns.
2. NaN handling diverges between backends: pandas treats NaN as null,
   polars treats NaN as a value. The same input can produce different
   drop vectors depending on backend. Narwhals exposes `fill_nan(None)`
   which would normalize this on numeric columns.
3. `n_unique()` materializes the full distinct set even though the rule
   only needs to know whether the count exceeds 1. Not a correctness
   concern but a possible perf optimization.

These are deferred because the current usage is internal and the inputs
are well-behaved; the issue tracks hardening for a future pass.
