# Plan: `step_dummy` and `step_one_hot` in `mlfeatures/`

## Context

Two separate steps exercise the `ONE_TO_MANY` mode defined in
`design/step-contract.md`. Both take categorical (and other discrete) columns
as input and expand each into binary indicator columns. They differ only in how
many columns they produce:

| Step | What it does |
|---|---|
| `step_dummy` | Drop-one (treatment) encoding: *k−1* columns per input |
| `step_one_hot` | Full indicator encoding: *k* columns per input |

The transform logic is identical between the two; it is extracted into a shared
utility function `apply_dummy` that lives in `step_dummy.py` and is imported by
`step_one_hot.py`.

---

## Files to Create / Modify

| File | Action |
|---|---|
| `mlfeatures/step_dummy.py` | **Create** — `fit_dummy`, `apply_dummy` (utility), `StepDummy` |
| `mlfeatures/step_one_hot.py` | **Create** — `fit_one_hot`, `StepOneHot` (calls `apply_dummy`) |
| `mlfeatures/__init__.py` | **Edit** — add `StepDummy`, `StepOneHot`, `fit_dummy`, `fit_one_hot` to `__all__` |
| `tests/test_step_dummy.py` | **Create** — unit tests for `fit_dummy` / `apply_dummy` / `StepDummy` |
| `tests/test_step_one_hot.py` | **Create** — unit tests for `fit_one_hot` / `StepOneHot` |
| `great-docs.yml` | **Edit** — add both steps to the `reference:` section |

---

## Design decisions

### Two classes, shared transform logic

`step_dummy` and `step_one_hot` are separate user-facing steps with distinct
names and behaviours. The fitting logic differs (one drops the reference level,
the other does not), so each has its own `fit_*` function. The transform logic
is identical — it takes a frame, a level mapping, and a naming function — so it
is extracted into `apply_dummy` in `step_dummy.py`. `StepOneHot.transform()`
imports and calls `apply_dummy` rather than duplicating the code.

### Accepted input types

`nw.String`, integer types (`nw.Int8` through `nw.Int128`, `nw.UInt8` through
`nw.UInt128`), `nw.Categorical`, `nw.Enum`, and `nw.Boolean` are all accepted.
Float and decimal types (`nw.Float32`, `nw.Float64`, `nw.Decimal`) are
rejected with a clear error — continuous numeric columns should not be
discretized implicitly by an encoding step.

### Column naming

Generated column names follow `{col}_{level}`. An optional
`name_fn: Callable[[str, str], str] | None` (default `None`) lets callers
override naming entirely — if provided it takes `(column_name, level)` and
returns the output column name.

Unlike R, we do **not** apply `make.names()`-style sanitization by default.
Spaces and other special characters in level names are preserved verbatim. If a
caller needs sanitized names they can supply a `name_fn`.

### Level ordering and stability

Levels are collected at `fit()` time via `.drop_nulls().unique().sort()` —
ascending lexicographic order — so output column order is deterministic
regardless of observation order in training data. The sorted level list is
stored on the step so `transform()` never re-examines data.

For `nw.Enum` (ordered categorical), sorting by declared order rather than
lexicographically is deferred to a later iteration; for v0.1 all columns sort
lexicographically.

### Drop-one logic (`step_dummy`)

The first level in the sorted list is the reference level and is dropped —
matching R's default `contr.treatment`. No `drop="first"/"last"` knob is
exposed for v0.1.

### Single-level columns

If a column has exactly one unique non-null level, the `fit_*` function raises
a `ValueError` at fit time. R raises this at bake/transform time, but catching
it earlier produces a better error message.

### Unknown levels at transform time

If `new_data` contains a level not seen during `fit()`, a **warning** is issued
and the corresponding rows in all indicator columns for that level are set to
null (not zero). This matches R's `warn_new_levels()` behaviour.

### Null handling

Null values in the source column produce null (not zero) in every indicator
column for that row — matching R.

### `keep_original_cols`

Default `False` (source columns are dropped after expansion). When `True`,
source columns are retained alongside the generated indicators.

### `accepts_dtypes()`

Both steps use the same accepted dtype set:

```python
def accepts_dtypes(self) -> AcceptedDtypes:
    return {
        nw.String,
        nw.Boolean,
        nw.Categorical,
        nw.Enum,
        nw.Int8, nw.Int16, nw.Int32, nw.Int64, nw.Int128,
        nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64, nw.UInt128,
    }
```

Float and decimal types (`nw.Float32`, `nw.Float64`, `nw.Decimal`) are
explicitly excluded — the framework's dtype check raises an error before
`fit()` runs.

### Output dtype

Indicators are cast to `nw.Int8` (0/1).

### Column name collision detection

If a generated indicator column name already exists in the frame at transform
time, raise a `ValueError` naming the conflicting column. R has the same check
via `check_name()`.

### `mode`

Both steps use `Mode.ONE_TO_MANY`.

### Contrasts (deferred)

R supports polynomial contrasts for ordered factors (`contr.poly`). Not
implemented in v0.1; ordered `nw.Enum` columns receive treatment-coding like
unordered categoricals.

### Sparse output (deferred)

Deferred to a later iteration — tracked in `design/ecosystem.md`.

---

## `fit_dummy` — implementation

### Signature

```python
def fit_dummy(
    features: nw.DataFrame[Any],
) -> dict[str, list[str]]:
```

- `features` — narwhals DataFrame whose columns are string, integer, boolean,
  categorical, or enum typed.
- Returns `dict[str, list[str]]` mapping each column name to its **retained**
  sorted levels (all levels minus the first, which is the reference).

### Algorithm

1. For each column:
   a. Collect non-null unique values: `.drop_nulls().unique().sort()` → `list[str]`.
   b. Raise `ValueError` if fewer than 2 unique levels.
   c. Remove the first element (reference level).
   d. Store the retained list in the result dict.
2. Return the dict.

---

## `fit_one_hot` — implementation

### Signature

```python
def fit_one_hot(
    features: nw.DataFrame[Any],
) -> dict[str, list[str]]:
```

Identical to `fit_dummy` except step (c) is omitted — all *k* levels are
retained. Raises `ValueError` the same way for single-level columns.

---

## `apply_dummy` — shared transform utility

Lives in `mlfeatures/step_dummy.py` and is imported by `mlfeatures/step_one_hot.py`.
Both `StepDummy.transform()` and `StepOneHot.transform()` delegate to it.

### Signature

```python
def apply_dummy(
    frame: Frame,
    levels: dict[str, list[str]],
    col_name_fn: Callable[[str, str], str],
    keep_original_cols: bool,
) -> Frame:
```

- `frame` — the internal narwhals frame to transform.
- `levels` — the stored level mapping from `fit_dummy` or `fit_one_hot`.
- `col_name_fn` — resolves `(source_col, level)` to the output column name.
- `keep_original_cols` — whether to retain source columns after expansion.

### Algorithm

For each source column `col` with retained level list `levels[col]`:

1. Detect unseen levels by comparing `frame[col].drop_nulls().unique()` against
   `levels[col]`; if any are found, emit a warning and those rows become null
   in all indicators.
2. For each retained level, append an indicator column:
   ```python
   frame = frame.with_columns(
       nw.when(nw.col(col) == level)
         .then(nw.lit(1, dtype=nw.Int8))
         .when(nw.col(col).is_null())
         .then(nw.lit(None, dtype=nw.Int8))
         .otherwise(nw.lit(0, dtype=nw.Int8))
         .alias(col_name_fn(col, level))
   )
   ```
3. After processing all levels for `col`, check that none of the generated
   column names collide with pre-existing non-source columns; raise `ValueError`
   if they do.
4. After all columns are processed, if `not keep_original_cols`, drop source
   columns.
5. Return the transformed frame.

---

## `StepDummy` class

```python
class StepDummy(Step):
    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: list[str],
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        id: str | None = None,
    ): ...
```

Private state populated by `fit()`:

- `_levels: dict[str, list[str]] | None`
- `_schema_in: ColSchema | None`
- `_schema_out: ColSchema | None`

### `fit()`

1. Select `self.columns` via `_frame.select()`.
2. Call `fit_dummy(features)` → `self._levels`.
3. Record `_schema_in` from selected column dtypes.
4. Record `_schema_out` by expanding `self._levels` through `_col_name()`.
5. Return `self`.

### `transform()`

```python
return apply_dummy(new_data, self._levels, self._col_name, self.keep_original_cols)
```

### `_col_name(col, level) -> str`

```python
def _col_name(self, col: str, level: str) -> str:
    if self.name_fn is not None:
        return self.name_fn(col, level)
    return f"{col}_{level}"
```

---

## `StepOneHot` class

Lives in `mlfeatures/step_one_hot.py`. Imports `apply_dummy` from `mlfeatures.step_dummy`.

```python
class StepOneHot(Step):
    mode = Mode.ONE_TO_MANY

    def __init__(
        self,
        columns: list[str],
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        id: str | None = None,
    ): ...
```

### `fit()`

Same as `StepDummy.fit()` but calls `fit_one_hot(features)`.

### `transform()`

```python
from mlfeatures.step_dummy import apply_dummy
return apply_dummy(new_data, self._levels, self._col_name, self.keep_original_cols)
```

`_col_name` is identical to `StepDummy._col_name`.

---

## `tests/test_step_dummy.py` — cases to cover

### `fit_dummy`

| Case | Expected |
|---|---|
| Two-level column | One level retained (the non-first) |
| Three-level column | *k−1* levels retained, sorted ascending |
| Null values in column | Nulls excluded from level list |
| Single-level column (after dropping nulls) | `ValueError` |
| Integer column | Levels sorted numerically; indicators named `{col}_{level}` |
| Boolean column | Two levels: `False` then `True`; `False` is reference, dropped |
| String column | Same behaviour as categorical |
| Return type | `dict[str, list[str]]` |

### `apply_dummy`

| Case | Expected |
|---|---|
| Known levels only | Correct indicators, no warning |
| Novel level in frame | Warning issued, that row is null in all indicators |
| Null rows in source column | Null propagated to every indicator for that row |
| Novel + null combined | Warning covers novel levels; null rows propagate independently |
| Column name collision | `ValueError` naming the colliding column |
| `keep_original_cols=True` | Source columns retained alongside indicators |
| `keep_original_cols=False` | Source columns dropped |
| 0-row frame | Schema-correct empty frame |
| 1-row frame | Correct single-row output |

### `StepDummy` — correctness

| Case | Expected |
|---|---|
| Basic fit+transform on training data | Correct *k−1* indicator columns, source col dropped |
| Multi-column input | Each column independently expanded |
| `keep_original_cols=True` | Source columns retained |
| Values correct for known fixed data | Indicator values match hand-computed expected matrix |
| Column names follow `{col}_{level}` | Names match expected list |
| `name_fn` overrides naming | Custom names used |

### `StepDummy` — guards and infrastructure

| Case | Expected |
|---|---|
| Single-level column | `ValueError` at fit time |
| `transform()` before `fit()` | `RuntimeError` |
| `schema_in()` before `fit()` | `RuntimeError` |
| `schema_out()` before `fit()` | `RuntimeError` |
| Pickle round-trip | `transform()` output identical |
| Pandas backend | Correct output, pandas frame returned |
| Polars backend | Correct output, polars frame returned |
| Empty column selection | Frame returned as-is, no error |
| `StepConformanceTest` | Passes all conformance assertions |

---

## `tests/test_step_one_hot.py` — cases to cover

### `fit_one_hot`

| Case | Expected |
|---|---|
| Two-level column | Both levels retained |
| Three-level column | All *k* levels retained, sorted ascending |
| Null values in column | Nulls excluded from level list |
| Single-level column | `ValueError` |
| Return type | `dict[str, list[str]]` |

### `StepOneHot` — correctness and infrastructure

All `StepDummy` correctness and infrastructure cases carry over, with *k*
indicator columns expected instead of *k−1*. The `apply_dummy` edge cases
(null, novel, collision, keep_original_cols, 0/1 rows) are not re-tested here
because they are covered by the `apply_dummy` unit tests in
`test_step_dummy.py`.

---

## Deviations from R `recipes`

| Behaviour | R `recipes` | This plan |
|---|---|---|
| `one_hot` as a separate step | No (it's a param of `step_dummy`) | Yes — `StepOneHot` is a separate class |
| Novel levels at transform | Warn + produce `NA` | Warn + produce null (equivalent) |
| Single-level column | Error at `bake()` time | Error at `fit()` time |
| Column name sanitization | `make.names()` applied by default | No sanitization; caller uses `name_fn` |
| String/int/bool input | Error: must be factor | Accepted |
| Float/decimal input | Error: must be factor | Error: float/decimal not accepted |
| Contrasts for ordered factors | `contr.poly` via `contrasts` arg | Deferred; ordered treated as unordered |
| Sparse output | `sparse="yes"/"no"/"auto"` | Deferred |

---

## Open questions

1. **`nw.Enum` level ordering** — should sorted levels for `nw.Enum` respect
   the declared enum order rather than lexicographic sort? Deferred for v0.1.

2. **`step_novel` / `step_unknown`** — the null-propagation for unseen levels
   motivates a `step_novel` preprocessing step. Out of scope here.

3. **Column name collision at fit vs transform** — detecting collisions at
   `fit()` time requires schema-propagation infrastructure not yet built. For
   now, detect at `transform()` time.

---

## Verification

```bash
uv run pytest tests/test_step_dummy.py tests/test_step_one_hot.py -v
uv run ruff check mlfeatures/step_dummy.py mlfeatures/step_one_hot.py
uv run ruff format --check mlfeatures/step_dummy.py mlfeatures/step_one_hot.py
uv run mypy mlfeatures/step_dummy.py mlfeatures/step_one_hot.py
uv run pyright mlfeatures/step_dummy.py mlfeatures/step_one_hot.py
```
