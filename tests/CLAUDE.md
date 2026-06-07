# tests

## Per-step test files

Each step function has its own series of test files. For a step `step_x`, there are three:

- `test_step_x.py` — unit tests for the step's own logic (fit/transform behavior, estimates, guards, edge cases).
- `test_step_x_e2e.py` — end-to-end tests exercising the step through a recipe.
- `test_step_x_infrastructure.py` — conformance/infrastructure tests (the `StepConformanceTest` contract: pickle round-trip, stable `id`, `produces_sparse`, dtype acceptance, schema_in/out, etc.).

## Infrastructure test coverage

Every `test_step_x_infrastructure.py` must contain **every** infrastructure test class — the set is uniform across all steps so the conformance contract is enforced identically everywhere. The current classes are:

- `TestEmptySelection` — an empty column selection is a no-op (frame passes through unchanged).
- `TestRowCountEdges` — `transform()` on 0-row and 1-row frames yields schema-correct output.
- `TestPickleRoundTrip` — a pickled fitted step reproduces the same `transform()` output.
- `TestFitIdempotence` — fitting twice on the same training data yields identical `transform()` output.
- `TestStableId` — the same step type with the same hyperparameters yields the same `id`.

If a class genuinely does not apply to a step, **still add the class** (or keep its slot visible) and write a comment explaining why instead of silently omitting it — so a reader can see the test was considered, not forgotten. For example, `TestEmptySelection` does not apply to `step_intercept` (an ADDING step with no column selector), so that file documents the omission in its module docstring rather than dropping the class without explanation.

When a new infrastructure test class is added, add it to **all** step files — including a no-op-with-comment slot for any step it does not apply to.

## Input dtype coverage

Steps that accept multiple input dtypes — like `step_normalize`, which accepts both floats and integers — must have unit tests verifying they work with each accepted dtype.

These same tests must also assert that the output dtype is always float, even when an integer output could happen to be produced by chance.
