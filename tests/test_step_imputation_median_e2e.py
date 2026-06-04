"""End-to-end lifecycle tests for StepImputationMedian: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepImputationMedian +
narwhals frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepImputationMedian


def _data() -> dict[str, list[float | None]]:
    return {
        "a": [1.0, None, 3.0, None, 100.0],  # median of observed (1,3,100) = 3.0
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    }


class TestEndToEnd:
    def test_pandas_imputes_nulls_with_median(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputationMedian(["a", "b"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1.0, 3.0, 3.0, 3.0, 100.0]
        assert out["b"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_polars_imputes_nulls_with_median(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepImputationMedian(["a"])).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1.0, 3.0, 3.0, 3.0, 100.0]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputationMedian(["a"])]).fit(df).transform(df)
        assert out["a"].tolist() == [1.0, 3.0, 3.0, 3.0, 100.0]
        assert out["b"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_transform_uses_train_median_not_new_data(self) -> None:
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})  # train median = 2.0
        fitted = Recipe(steps=[StepImputationMedian(["a"])]).fit(train)
        new = pd.DataFrame({"a": [None, 100.0]})
        out = fitted.transform(new)
        # null filled with TRAIN median (2.0), not new-data median.
        assert out["a"].tolist() == [2.0, 100.0]

    def test_integer_with_nulls_promoted_to_float(self) -> None:
        df = pl.DataFrame({"a": [1, None, 3]})  # Int64 with a null
        out = Recipe(steps=[StepImputationMedian(["a"])]).fit(df).transform(df)
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_all_null_column_warns_and_passes_through(self) -> None:
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "empty": pd.Series([None, None, None], dtype="float64"),
            }
        )
        recipe = Recipe(steps=[StepImputationMedian(["a", "empty"])])
        with pytest.warns(UserWarning, match="entirely null"):
            fitted = recipe.fit(df)
        out = fitted.transform(df)
        assert out["a"].tolist() == [1.0, 2.0, 3.0]
        assert out["empty"].isna().all()  # untouched

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepImputationMedian(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepImputationMedian(["a"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepImputationMedianUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepImputationMedian(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepImputationMedian(["a"]).id == "imputation_median"

    def test_explicit_id(self) -> None:
        assert StepImputationMedian(["a"], id="impute_med").id == "impute_med"

    def test_fit_returns_self(self) -> None:
        step = StepImputationMedian(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputationMedian(["a"]).transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputationMedian(["a"]).schema_in()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepImputationMedian(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_imputed_columns_float(self) -> None:
        step = StepImputationMedian(["a"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_out().dtypes["a"] == "Float64"

    def test_not_sparse(self) -> None:
        assert StepImputationMedian(["a"]).produces_sparse is False
