"""Conformance/infrastructure tests for StepIntercept.

Part of the per-step ``StepConformanceTest`` contract (see
``design/step-contract.md``): the checks here are generic properties every step
must satisfy, not step-specific behavior.

Note: the empty-selection conformance check does not apply to StepIntercept.
It is an ADDING step that takes no column selector — it always appends a single
constant column regardless of the input columns — so there is no "zero features
selected" state to exercise.
"""

from __future__ import annotations
