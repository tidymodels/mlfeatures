# Frame module — coverage notes

→ See also: [ecosystem.md](ecosystem.md)

---

Findings from building `mlfeatures/_frame.py` (the narwhals boundary) and prototyping
`step_normalize`'s data ops over pandas and polars (issue #15). This is the
"document any coverage gaps" deliverable — the early-warning signal for the
fallback to a concrete internal data frame type.

## Verdict

**Narwhals is thick enough for the `step_normalize` workload.** The full data-ops
core — numeric-column selection, per-column `mean`/`std`, and a
`(x - mean) / std` `with_columns` transform — was written **once** against
`narwhals.stable.v1` and produced identical results on pandas and polars, with the
input backend preserved on output. **No `to_native` escape hatch was needed.**
Boundary, column-selection, type-aware, and categorical-detection helpers likewise
work uniformly across both backends. The riskier probes (categoricals via
`step_unknown`, row ops via `step_filter`, sparse output via `step_dummy`) are
deferred to the prototype step set [#18]; nothing here contradicts the v0.1
decision.

## Gaps and sharp edges

These are not blockers, but step authors and #18 should know them.

- **`narwhals.typing.IntoDataFrame` is not pyright-strict clean.** Its definition
  references an optional DuckDB native type that resolves as `Unknown` under
  `pyright` strict mode, producing `reportUnknownParameterType` errors at the
  boundary functions. We sidestepped this by typing `from_native` / `to_native`
  against the project's own native `DataFrame` union (`pandas | polars`) rather
  than narwhals' `IntoDataFrame`. Revisit if/when we widen the accepted input
  types beyond pandas/polars, or if narwhals tightens that stub.

- **NaN vs. null is backend-dependent.** `null_count()` (backing
  `_frame.null_counts` / `has_nulls`) counts NumPy `NaN` **as null on pandas**, but
  polars treats `NaN` and `null` as distinct (a float `NaN` is *not* counted as
  null). So `has_nulls` does not mean the same thing on both backends for float
  columns containing `NaN`. Steps with an explicit NA policy (per
  [ecosystem.md](ecosystem.md) § Missing values) must account for this; a future
  `_frame` helper may need to expose `is_nan` separately from `is_null`.

- **`std` defaults to `ddof=1` on both backends** — narwhals normalizes this, so
  pandas and polars agree out of the box. Worth noting because sample vs.
  population standard deviation is a classic place backend defaults diverge; the
  eventual `step_normalize` should still make `ddof` explicit rather than relying
  on the default.

- **`from_native(..., eager_only=True)`** is used throughout — the data frame API is
  eager-only for now. Lazy / streaming data frames are out of scope until a step needs
  them; that will reopen the boundary signature.

## Tooling note (not a narwhals gap)

`uv sync` installs only the default dependency group, so the type-checkers and
linters in the `[dev]` *optional extra* were not installed by a plain `uv sync` —
`uv run mypy` silently fell back to a global shim. Use `uv sync --extra dev` (or
move these tools into a dependency group) so `uv run mypy` / `uv run pyright` run
against the project venv as the CI workflow expects.
