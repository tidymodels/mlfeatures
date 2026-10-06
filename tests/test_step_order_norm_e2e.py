"""End-to-end lifecycle tests for StepOrderNorm: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepOrderNorm + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepOrderNorm


def _data() -> dict[str, list[float]]:
    # ``const`` is degenerate (a single distinct value); ``a`` and ``b`` vary.
    return {
        "a": [1.0, 2.0, 4.0, 8.0, 16.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_reshapes_to_normal_quantiles(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOrderNorm(["a", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "const", "b"]  # structure preserved
        # A monotone increasing input maps to a monotone increasing output.
        assert out["a"].is_monotonic_increasing

    def test_polars_reshapes_to_normal_quantiles(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepOrderNorm(["a", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert out["a"].is_sorted()

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOrderNorm(["a"])])
        out = recipe.fit(df).transform(df)
        assert out["b"].tolist() == [5.0, 1.0, 4.0, 1.0, 9.0]
        assert out["const"].tolist() == [7.0] * 5

    def test_degenerate_column_warns_and_is_left_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOrderNorm(["a", "const"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        out = fitted.transform(df)
        assert out["const"].tolist() == [7.0] * 5

    def test_transform_uses_train_lookup_not_new_data(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        fitted = Recipe(steps=[StepOrderNorm(["a"])]).fit(train)
        new = pd.DataFrame({"a": [3.0]})
        out = fitted.transform(new)
        # 3.0 is the training median -> maps to 0, regardless of "new" framing.
        assert out["a"].tolist() == pytest.approx([0.0], abs=1e-9)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepOrderNorm(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        out = Recipe(steps=[StepOrderNorm(["a"])]).fit(df).transform(df)
        assert out["a"].dtype == float

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepOrderNorm(["a", "b"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepOrderNormUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepOrderNorm(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepOrderNorm(["a"]).id == "order_norm"

    def test_explicit_id(self) -> None:
        assert StepOrderNorm(["a"], id="my_orq").id == "my_orq"

    def test_fit_returns_self(self) -> None:
        step = StepOrderNorm(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepOrderNorm(["a"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepOrderNorm(["a"]).schema_in()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepOrderNorm(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_normalized_columns_float(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "c": [7, 7, 7, 7]})
        step = StepOrderNorm(["a", "c"])
        with pytest.warns(UserWarning):
            step.fit(nw.from_native(df), SchemaInfo())
        out_schema = step.schema_out()
        assert out_schema.dtypes["a"] == "Float64"
        assert out_schema.dtypes["c"] == "Int64"  # degenerate int kept as-is

    def test_required_pkgs(self) -> None:
        assert StepOrderNorm(["a"]).required_pkgs() == ["numpy", "scipy"]

    def test_not_sparse(self) -> None:
        assert StepOrderNorm(["a"]).produces_sparse is False
