# `step_intercept` — add a constant (fixed-value) column

## Goal

The `ADDING`-mode prototype from `design/steps-catalog.md`: add a constant
column. Generalised from R `recipes`' "column of ones" to a **configurable fixed
value** (default `1.0`).

## Approach

Follows the class-based examples now on `main` — `StepDropZeroVariance` (PR #52)
and `StepDropCorr` — on top of the `Recipe`/`FittedRecipe` lifecycle (PR #48).
Each step module pairs a plain logic function with a thin `Step` subclass that
wraps it in the fit/transform lifecycle.

An intercept estimates nothing, so (unlike the REMOVING filters) it has no
`fit_*` half: the logic function is the *apply* half, and `fit()` only records
the produced column's schema.

## Public surface — `mlfeatures/step_intercept.py`

```python
add_intercept(features: Frame, value: float = 1.0, name: str = "intercept") -> Frame
```
Appends one constant column via `features.with_columns(nw.lit(value).alias(name))`;
raises `ValueError` if `name` already exists (rather than silently overwriting).

```python
class StepIntercept(Step):   # mode = Mode.ADDING
    StepIntercept(value=1.0, name="intercept", role="predictor", id=None)
```
- `fit()` — builds the column on the training frame to capture the produced
  dtype (and surface a name clash early); records `schema_out`. Estimates nothing.
- `transform()` — delegates to `add_intercept`; `RuntimeError` if called before fit.
- `schema_in()` — empty (consumes no columns); `schema_out()` — the one produced column.
- `accepts_dtypes()` — `ANY_DTYPE` (no inputs to constrain); `produces_sparse` — `False`.
- `output_role` carries `role` (default `"predictor"`) — the first column-producing
  step, so the contract's new-column role applies here.

No `Recipe.step_*()` sugar exists; users compose via `Recipe(steps=[StepIntercept()])`
or `Recipe().add(StepIntercept())`.

## Wiring
- Export `StepIntercept` from `mlfeatures/__init__.py` `__all__` (mirrors the other steps).
- Add `StepIntercept` to the `reference:` Classes list in `great-docs.yml` (PR #52 example).

## Tests
- `tests/test_intercept.py` — unit tests for `add_intercept` (return type, core
  behavior, edge shapes, pandas+polars, name-clash `ValueError`).
- `tests/test_step_intercept_e2e.py` — full `Recipe → fit → transform` on pandas
  and polars, plus the step contract surface (mode, id, output_role, schema_in/out,
  accepts_dtypes, transform-before-fit guard, fit-time clash error).

## Out of scope
No `Step` content-hash id, selectors, lineage DAG, or `StepConformanceTest`
(framework-wide follow-ups, consistent with the other merged steps).
