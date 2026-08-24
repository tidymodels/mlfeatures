# Non-goals, Open Questions, and Success Criteria

→ See also: [steps-catalog.md](steps-catalog.md), [lifecycle.md](lifecycle.md), [serialization.md](serialization.md)

---

## Non-goals for v1

- **Not a modeling framework.** Recipes produces feature matrices; model fitting is downstream (sklearn, xgboost, torch, etc.).
- **Not a general dataframe DSL.** `step_mutate` exists for convenience but the package does not aim to replace pandas/polars.
- **Not a distributed compute engine.** In-memory only at v1; Dask/Spark backends are future work — and explicitly tabled, not accidentally omitted.
- **Not a tuner.** Hyperparameter sweeps are delegated to an external tune package (analogue of R `tune`).
- **No `untransform()` / inverse transforms in core.** Repeatedly rejected in R; belongs in workflow post-processing because the model — not the recipe — owns the prediction surface [#264, #998].
- **No in-recipe lagging that could leak.** Pre-lag before the recipe; doing it inside risks leakage [#583].
- **No groupwise generic steps.** Misuse risk too high [#630]; offer specific grouped steps where the math makes sense.
- **No arbitrary higher-order "custom transformation" steps beyond `step_mutate`.** `step_mutate` is the maximum concession [#224].
- **No embedding pretrained-model pointers in steps.** Pointers don't survive serialization [#393]; relevant for steps wrapping embeddings / transformers — explicit design constraint.
- **No lazy/remote/database execution.** A deliberate v2 decision, not an oversight [#1007].

---

## Open questions

Decisions that must be made **before v0.1 ships**.

1. **Package name.** `shoez` is the current working name — confirm it is available on PyPI and reserve it early.

2. **Pandas vs. polars as primary boundary.** ~~Pick one as the canonical external-facing type; the other becomes an adapter.~~ **Resolved: narwhals is the internal representation** (see [ecosystem.md](ecosystem.md#dataframe-support)). Steps are written once against the narwhals API and run natively on whichever backend the user passes; no single internal frame type is committed to. Pandas/numpy remains the canonical type at the sklearn boundary.

4. **Lazy vs. eager transform.** Should `transform()` return a lazy polars frame to enable query optimization? Recommendation: **eager by default**; add `transform_lazy()` later. This must be decided explicitly — R `recipes` never did [#878].

5. **Step naming convention.** Keep R's `step_dummy` function-based API (recommended for parity) or switch to class-based (`Dummy`, `Normalize`)? Recommendation: **functions named `step_dummy(...)` returning `Step` instances**.

6. **Deprecation policy.** "One minor version of warning, removal in the next major" vs. a longer cycle. Must be documented in v0.1 release notes.

7. **Multi-column structured targets.** Survival targets, multi-output regression, and multitask learning need targets as a structured object (or a role applied to multiple columns), not a single formula LHS. First-class representation vs. emergent from formula tricks — decide before the role API is finalized.
8. **Anti-leakage guardrails.** At minimum: warn when a step's `fit()` consumes the target column (other than target-encoding steps designed to do so). Research item for a richer solution [#414].

9. **Mutating vs. immutable `.add()`.** When steps are appended to a recipe via a builder method, does `.add()` mutate the recipe in place (as the [gist prototype](https://gist.github.com/EmilHvitfeldt/8b134755222a1ba43f3ee8fdf7461b31) does, returning `None`) or return a new `Recipe`? Returning a new recipe is consistent with the [lifecycle.md](lifecycle.md) "deferred, inspectable, modifiable before fitting" stance and lets chains compose, but costs a copy per call. The `steps=[...]` constructor form sidesteps the question and is the recommended primary spelling regardless. Recommendation: **`.add()` returns a new `Recipe`.** See [interfaces.md](interfaces.md).

---

## v0.1 Success criteria

- All Tier 1 steps and checks implemented with parity tests against R `recipes` (same fit/transform output on shared fixture datasets, within numerical tolerance).
- Selector DSL fully working.
- Fit/transform/tidy/serialize lifecycle round-trips.
- Sklearn `Pipeline` adapter usable end-to-end on a real dataset (e.g. Ames housing or credit_data).
- `StepConformanceTest` passes for every shipped step. See [step-contract.md](step-contract.md).
- Docs: an "Introduction" notebook, a "Selectors" notebook, a "Roles" notebook, a "Writing your own step" notebook — mirroring the existing R vignettes.
