"""Tests for the ptype snapshot + boundary validation. See ``design/lifecycle.md``.

A fitted recipe snapshots the training schema (names, dtypes, roles) into a
``Ptype`` and validates new data against it at ``transform()``: required columns
must be present and present columns must match the training dtype. Per-role
requirements decide *which* columns are required at transform time, and
``Recipe.update_role_requirements`` overrides them.
"""

from __future__ import annotations

import pickle

import pandas as pd
import polars as pl
import pytest

from shoez import Ptype, Recipe, Selectors, StepNormalize


def _train() -> dict[str, list[float]]:
    return {"x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 0.0, 1.0]}


class TestTransformColumnPresence:
    def test_missing_predictor_errors(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError, match="missing required column.*'x'"):
            fitted.transform(pd.DataFrame({"y": [0.0, 1.0]}))

    def test_missing_outcome_is_allowed(self) -> None:
        # The outcome is never required at transform time.
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0]}))
        assert list(out.columns) == ["x"]

    def test_extra_columns_ignored(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0], "extra": [9.0, 9.0]}))
        assert "x" in out.columns


class TestDtypeMatching:
    def test_dtype_mismatch_errors(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError, match="dtype mismatch.*'x'"):
            fitted.transform(pd.DataFrame({"x": ["a", "b"]}))

    def test_matching_dtype_ok(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        out = fitted.transform(pd.DataFrame({"x": [10.0, 20.0]}))
        assert list(out.columns) == ["x"]


class TestRoleRequirements:
    def test_custom_role_required_by_default(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(predictors="x")
        rec.roles.add("y", "id")
        fitted = rec.add(StepNormalize(["x"])).fit(df)
        assert fitted.ptype is not None
        assert fitted.ptype.transform_required["y"] is True

    def test_update_role_requirements_by_role_name(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(predictors="x")
        rec.roles.add("y", "id")
        fitted = (
            rec.update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x"]))
            .fit(df)
        )
        assert fitted.ptype is not None
        assert fitted.ptype.transform_required["y"] is False
        # And the now-optional column may be dropped at transform.
        out = fitted.transform(pd.DataFrame({"x": [1.0, 2.0]}))
        assert list(out.columns) == ["x"]

    def test_update_role_requirements_by_selector(self) -> None:
        df = pd.DataFrame(_train())
        rec = Recipe(predictors="x")
        rec.roles.add("y", "id")
        fitted = (
            rec.update_role_requirements(Selectors.col("y"), transform_required=False)
            .add(StepNormalize(["x"]))
            .fit(df)
        )
        assert fitted.ptype is not None
        assert fitted.ptype.transform_required["y"] is False

    def test_predictor_requirement_not_configurable(self) -> None:
        with pytest.raises(ValueError, match="not configurable"):
            Recipe(predictors="x").update_role_requirements(
                "predictor", transform_required=False
            )

    def test_outcome_requirement_not_configurable(self) -> None:
        with pytest.raises(ValueError, match="not configurable"):
            Recipe(outcome="y").update_role_requirements(
                "outcome", transform_required=True
            )

    def test_update_role_requirements_is_pure(self) -> None:
        rec = Recipe(predictors="x")
        rec.roles.add("y", "id")
        new = rec.update_role_requirements("id", transform_required=False)
        assert new is not rec
        assert rec._role_req_overrides == []


class TestPolarsBackend:
    def test_missing_predictor_errors(self) -> None:
        df = pl.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError, match="missing required column"):
            fitted.transform(pl.DataFrame({"y": [0.0, 1.0]}))

    def test_dtype_mismatch_errors(self) -> None:
        df = pl.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        with pytest.raises(ValueError, match="dtype mismatch"):
            fitted.transform(pl.DataFrame({"x": ["a", "b"]}))


class TestPickle:
    def test_ptype_survives_pickle_roundtrip(self) -> None:
        df = pd.DataFrame(_train())
        fitted = Recipe(outcome="y", predictors="x").add(StepNormalize(["x"])).fit(df)
        restored = pickle.loads(pickle.dumps(fitted))
        assert isinstance(restored.ptype, Ptype)
        assert restored.ptype.transform_required == fitted.ptype.transform_required
        # The restored recipe still enforces the boundary check.
        with pytest.raises(ValueError, match="missing required column"):
            restored.transform(pd.DataFrame({"y": [0.0]}))
