"""Tests for the schema snapshot + boundary validation. See ``design/lifecycle.md``.

A fitted recipe snapshots the training schema (names, dtypes, roles) into a
``Schema`` and validates new data against it at ``transform()``: required columns
must be present and present columns must match the training dtype. Per-role
requirements decide *which* columns are required at transform time, and
``Recipe.update_role_requirements`` overrides them.
"""

from __future__ import annotations

import pickle

import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Recipe, Schema, Selectors, StepNormalize


def _train() -> dict[str, list[float]]:
    return {"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 0.0, 1.0]}


class TestTransformColumnPresence:
    def test_missing_feature_errors(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(pd.DataFrame({"y": [0.0, 1.0]}))
        assert str(excinfo.value) == snapshot

    def test_missing_target_is_allowed(self) -> None:
        # The target is never required at transform time.
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0]}))
        assert list(out.columns) == ["x"]

    def test_extra_columns_ignored(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0], "extra": [9.0, 9.0]}))
        assert "x" in out.columns


class TestDtypeMatching:
    def test_dtype_mismatch_errors(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(pd.DataFrame({"x": ["a", "b"]}))
        assert str(excinfo.value) == snapshot

    def test_matching_dtype_ok(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [10.0, 20.0]}))
        assert list(out.columns) == ["x"]


class TestRoleRequirements:
    def test_custom_role_required_by_default(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(features="x")
        rec.roles.add("y", "id")
        fitted = rec.add(StepNormalize(["x"])).fit(df)
        assert fitted.schema is not None
        assert fitted.schema.transform_required["y"] is True

    def test_update_role_requirements_by_role_name(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(features="x")
        rec.roles.add("y", "id")
        fitted = (
            rec.update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x"]))
            .fit(df)
        )
        assert fitted.schema is not None
        assert fitted.schema.transform_required["y"] is False
        # And the now-optional column may be dropped at transform.
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0]}))
        assert list(out.columns) == ["x"]

    def test_update_role_requirements_by_selector(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(features="x")
        rec.roles.add("y", "id")
        fitted = (
            rec.update_role_requirements(Selectors.col("y"), transform_required=False)
            .add(StepNormalize(["x"]))
            .fit(df)
        )
        assert fitted.schema is not None
        assert fitted.schema.transform_required["y"] is False

    def test_feature_requirement_not_configurable(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(features="x").update_role_requirements(
                "feature", transform_required=False
            )
        assert str(excinfo.value) == snapshot

    def test_target_requirement_not_configurable(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(ValueError) as excinfo:
            Recipe(target="y").update_role_requirements(
                "target", transform_required=True
            )
        assert str(excinfo.value) == snapshot

    def test_update_role_requirements_is_pure(self) -> None:
        rec = Recipe(features="x")
        rec.roles.add("y", "id")
        new = rec.update_role_requirements("id", transform_required=False)
        assert new is not rec
        assert rec._role_req_overrides == []


class TestPolarsBackend:
    def test_missing_feature_errors(self, snapshot: SnapshotAssertion) -> None:
        df = pl.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(pl.DataFrame({"y": [0.0, 1.0]}))
        assert str(excinfo.value) == snapshot

    def test_dtype_mismatch_errors(self, snapshot: SnapshotAssertion) -> None:
        df = pl.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError) as excinfo:
            fitted.transform(pl.DataFrame({"x": ["a", "b"]}))
        assert str(excinfo.value) == snapshot


class TestPickle:
    def test_schema_survives_pickle_roundtrip(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(target="y", features="x").add(StepNormalize(["x"])).fit(df)
        restored = pickle.loads(pickle.dumps(fitted))
        assert isinstance(restored.schema, Schema)
        assert restored.schema.transform_required == fitted.schema.transform_required
        # The restored recipe still enforces the boundary check.
        with pytest.raises(ValueError) as excinfo:
            restored.transform(pd.DataFrame({"y": [0.0]}))
        assert str(excinfo.value) == snapshot
