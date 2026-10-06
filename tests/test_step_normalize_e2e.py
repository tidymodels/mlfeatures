"""End-to-end lifecycle tests for StepNormalize: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepNormalize + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepNormalize


def _data() -> dict[str, list[float]]:
    # ``const`` is zero-variance; ``a`` and ``b`` vary.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_centers_and_scales(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepNormalize(["a", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "const", "b"]  # structure preserved
        for col in ("a", "b"):
            assert out[col].mean() == pytest.approx(0.0)
            assert out[col].std(ddof=1) == pytest.approx(1.0)

    def test_polars_centers_and_scales(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepNormalize(["a", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        for col in ("a", "b"):
            assert out[col].mean() == pytest.approx(0.0)
            assert out[col].std(ddof=1) == pytest.approx(1.0)

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        # Only normalize "a"; "const" and "b" must pass through untouched.
        recipe = Recipe(steps=[StepNormalize(["a"])])
        out = recipe.fit(df).transform(df)
        assert out["b"].tolist() == [5.0, 1.0, 4.0, 1.0, 9.0]
        assert out["const"].tolist() == [7.0] * 5

    def test_constant_column_warns_and_is_left_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepNormalize(["a", "const"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        out = fitted.transform(df)
        # "const" is dropped from normalization -> passes through unchanged.
        assert out["const"].tolist() == [7.0] * 5
        assert out["a"].mean() == pytest.approx(0.0)

    def test_transform_uses_train_stats_not_new_data(self) -> None:
        train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})  # mean 2, std (ddof=1) 2
        fitted = Recipe(steps=[StepNormalize(["a"])]).fit(train)
        new = pd.DataFrame({"a": [2.0, 6.0]})
        out = fitted.transform(new)
        # (2-2)/2 = 0 ; (6-2)/2 = 2 — using TRAIN stats, not new-data stats.
        assert out["a"].tolist() == pytest.approx([0.0, 2.0])

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepNormalize(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4]})  # integer column
        out = Recipe(steps=[StepNormalize(["a"])]).fit(df).transform(df)
        assert out["a"].dtype == float  # promoted to float by normalization

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepNormalize(["a", "b"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepNormalizeUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepNormalize(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepNormalize(["a"]).id == "normalize"

    def test_explicit_id(self) -> None:
        assert StepNormalize(["a"], id="my_norm").id == "my_norm"

    def test_fit_returns_self(self) -> None:
        step = StepNormalize(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepNormalize(["a"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepNormalize(["a"]).schema_in()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepNormalize(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_normalized_columns_float(self) -> None:
        # "c" is a constant *integer* column: not normalized, keeps Int64.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": [7, 7, 7]})
        step = StepNormalize(["a", "c"])
        with pytest.warns(UserWarning):
            step.fit(nw.from_native(df), SchemaInfo())
        out_schema = step.schema_out()
        assert out_schema.dtypes["a"] == "Float64"  # normalized -> float
        assert out_schema.dtypes["c"] == "Int64"  # constant int kept as-is

    def test_not_sparse(self) -> None:
        assert StepNormalize(["a"]).produces_sparse is False
