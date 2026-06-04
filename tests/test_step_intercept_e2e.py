"""End-to-end lifecycle tests for StepIntercept: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepIntercept + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepIntercept


def _data() -> dict[str, list[float]]:
    return {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}


class TestEndToEnd:
    def test_pandas_appends_intercept(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepIntercept()])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b", "intercept"]
        assert out["intercept"].tolist() == [1.0, 1.0, 1.0]

    def test_polars_appends_intercept(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepIntercept(value=0.0, name="b0"))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["a", "b", "b0"]
        assert out["b0"].to_list() == [0.0, 0.0, 0.0]

    def test_original_columns_preserved(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepIntercept()]).fit(df).transform(df)
        assert out["a"].tolist() == [1.0, 2.0, 3.0]
        assert out["b"].tolist() == [4.0, 5.0, 6.0]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepIntercept()]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepIntercept()]).fit(df)
        assert all(step.trained for step in fitted.steps)

    def test_clash_with_existing_column_raises(self) -> None:
        df = pd.DataFrame({"intercept": [1.0, 2.0], "x": [3.0, 4.0]})
        with pytest.raises(ValueError, match="already exists"):
            Recipe(steps=[StepIntercept()]).fit(df)


class TestStepIntercept:
    def test_mode_is_adding(self) -> None:
        assert StepIntercept().mode is Mode.ADDING

    def test_default_id(self) -> None:
        assert StepIntercept().id == "intercept"

    def test_default_output_role_is_predictor(self) -> None:
        assert StepIntercept().output_role == "predictor"

    def test_custom_output_role(self) -> None:
        assert StepIntercept(role=None).output_role is None

    def test_fit_returns_self(self) -> None:
        step = StepIntercept()
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepIntercept()
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_out_describes_produced_column(self) -> None:
        step = StepIntercept()
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        schema = step.schema_out()
        assert schema.columns == ["intercept"]

    def test_schema_in_is_empty(self) -> None:
        # ADDING consumes no columns.
        assert StepIntercept().schema_in().columns == []

    def test_accepts_any_dtype(self) -> None:
        accepted = StepIntercept().accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.String in accepted

    def test_does_not_produce_sparse(self) -> None:
        assert StepIntercept().produces_sparse is False
