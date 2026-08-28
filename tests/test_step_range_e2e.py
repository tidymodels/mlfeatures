"""End-to-end lifecycle tests for StepRange: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepRange + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, Recipe, SchemaInfo, StepRange


def _data() -> dict[str, list[float]]:
    # ``const`` has a zero range; ``a`` and ``b`` vary.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "const": [7.0, 7.0, 7.0, 7.0, 7.0],
        "b": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_scales_to_unit_interval(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepRange(["a", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "const", "b"]  # structure preserved
        for col in ("a", "b"):
            assert out[col].min() == pytest.approx(0.0)
            assert out[col].max() == pytest.approx(1.0)
        assert out["a"].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_polars_scales_to_unit_interval(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepRange(["a", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        for col in ("a", "b"):
            assert out[col].min() == pytest.approx(0.0)
            assert out[col].max() == pytest.approx(1.0)

    def test_custom_range_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe().add(StepRange(["a"], min=-1.0, max=1.0))
        out = recipe.fit(df).transform(df)
        assert out["a"].tolist() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepRange(["a"])])
        out = recipe.fit(df).transform(df)
        assert out["b"].tolist() == [5.0, 1.0, 4.0, 1.0, 9.0]
        assert out["const"].tolist() == [7.0] * 5

    def test_constant_column_warns_and_is_left_unchanged(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepRange(["a", "const"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        out = fitted.transform(df)
        # "const" is dropped from the rescaling -> passes through unchanged.
        assert out["const"].tolist() == [7.0] * 5
        assert out["a"].min() == pytest.approx(0.0)

    def test_transform_uses_train_bounds_not_new_data(self) -> None:
        train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})  # bounds (0, 4)
        fitted = Recipe(steps=[StepRange(["a"])]).fit(train)
        new = pd.DataFrame({"a": [1.0, 3.0]})
        out = fitted.transform(new)
        # Using TRAIN bounds: 1/4 and 3/4, not a fresh 0..1 rescale of the new data.
        assert out["a"].tolist() == pytest.approx([0.25, 0.75])

    def test_out_of_range_values_are_clipped(self) -> None:
        train = pd.DataFrame({"a": [0.0, 4.0]})
        fitted = Recipe(steps=[StepRange(["a"])]).fit(train)
        out = fitted.transform(pd.DataFrame({"a": [-8.0, 2.0, 12.0]}))
        assert out["a"].tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_clipping_false_extrapolates(self) -> None:
        train = pd.DataFrame({"a": [0.0, 4.0]})
        fitted = Recipe(steps=[StepRange(["a"], clipping=False)]).fit(train)
        out = fitted.transform(pd.DataFrame({"a": [-8.0, 2.0, 12.0]}))
        assert out["a"].tolist() == pytest.approx([-2.0, 0.5, 3.0])

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepRange(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4]})  # integer column
        out = Recipe(steps=[StepRange(["a"])]).fit(df).transform(df)
        assert out["a"].dtype == float  # promoted to float by the rescaling

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepRange(["a", "b"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepRangeUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepRange(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepRange(["a"]).id == "range"

    def test_explicit_id(self) -> None:
        assert StepRange(["a"], id="my_range").id == "my_range"

    def test_fit_returns_self(self) -> None:
        step = StepRange(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepRange(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_rescaled_columns_float(self) -> None:
        # "c" is a constant *integer* column: not rescaled, keeps Int64.
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "c": [7, 7, 7]})
        step = StepRange(["a", "c"])
        with pytest.warns(UserWarning):
            step.fit(nw.from_native(df), SchemaInfo())
        out_schema = step.schema_out()
        assert out_schema.dtypes["a"] == "Float64"  # rescaled -> float
        assert out_schema.dtypes["c"] == "Int64"  # constant int kept as-is

    def test_not_sparse(self) -> None:
        assert StepRange(["a"]).produces_sparse is False
