# tests

## Per-step test files

Each step function has its own series of test files. For a step `step_x`, there are three:

- `test_step_x.py` — unit tests for the step's own logic (fit/transform behavior, estimates, guards, edge cases).
- `test_step_x_e2e.py` — end-to-end tests exercising the step through a recipe.
- `test_step_x_infrastructure.py` — conformance/infrastructure tests (the `StepConformanceTest` contract: pickle round-trip, stable `id`, `produces_sparse`, dtype acceptance, schema_in/out, etc.).

## Input dtype coverage

Steps that accept multiple input dtypes — like `step_normalize`, which accepts both floats and integers — must have unit tests verifying they work with each accepted dtype.

These same tests must also assert that the output dtype is always float, even when an integer output could happen to be produced by chance.
