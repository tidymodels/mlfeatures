"""End-to-end lifecycle tests for StepImputeMean: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepImputeMean + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepImputeMean


def _data() -> dict[str, list[float | None]]:
    return {
        "a": [1.0, None, 3.0, None, 5.0],  # mean of observed = 3.0
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    }


class TestEndToEnd:
    def test_pandas_imputes_nulls_with_mean(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputeMean(["a", "b"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out["a"].tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]
        assert out["b"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_polars_imputes_nulls_with_mean(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepImputeMean(["a"])).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out["a"].to_list() == [1.0, 3.0, 3.0, 3.0, 5.0]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputeMean(["a"])]).fit(df).transform(df)
        assert out["a"].tolist() == [1.0, 3.0, 3.0, 3.0, 5.0]
        assert out["b"].tolist() == [10.0, 20.0, 30.0, 40.0, 50.0]

    def test_transform_uses_train_mean_not_new_data(self) -> None:
        train = pd.DataFrame({"a": [0.0, 2.0, 4.0]})  # train mean = 2.0
        fitted = Recipe(steps=[StepImputeMean(["a"])]).fit(train)
        new = pd.DataFrame({"a": [None, 100.0]})
        out = fitted.transform(new)
        # null filled with TRAIN mean (2.0), not new-data mean.
        assert out["a"].tolist() == [2.0, 100.0]

    def test_integer_with_nulls_promoted_to_float(self) -> None:
        df = pl.DataFrame({"a": [1, None, 3]})  # Int64 with a null
        out = Recipe(steps=[StepImputeMean(["a"])]).fit(df).transform(df)
        assert out["a"].to_list() == [1.0, 2.0, 3.0]

    def test_all_null_column_warns_and_passes_through(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # 'empty' is typed float explicitly: an untyped all-null pandas column
        # is object-backed, which narwhals reports as String and the recipe's
        # dtype check then (correctly) rejects for a numeric-only step.
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "empty": pd.Series([None, None, None], dtype="float64"),
            }
        )
        recipe = Recipe(steps=[StepImputeMean(["a", "empty"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        # numpy may also emit a RuntimeWarning ("Mean of empty slice"); select
        # our UserWarning explicitly rather than relying on record order.
        assert str(record.pop(UserWarning).message) == snapshot
        out = fitted.transform(df)
        assert out["a"].tolist() == [1.0, 2.0, 3.0]
        assert out["empty"].isna().all()  # untouched

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepImputeMean(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepImputeMean(["a"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepImputeMeanUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepImputeMean(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepImputeMean(["a"]).id == "impute_mean"

    def test_explicit_id(self) -> None:
        assert StepImputeMean(["a"], id="impute").id == "impute"

    def test_fit_returns_self(self) -> None:
        step = StepImputeMean(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputeMean(["a"]).transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputeMean(["a"]).schema_in()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepImputeMean(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_imputed_columns_float(self) -> None:
        step = StepImputeMean(["a"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_out().dtypes["a"] == "Float64"

    def test_schema_out_promotes_selected_no_null_column(self) -> None:
        # A selected integer column with no nulls is still promoted to Float64
        # in schema_out (it has a valid mean), matching transform-time behavior.
        df = nw.from_native(pd.DataFrame({"a": [1, 2, 3, 4]}), eager_only=True)
        step = StepImputeMean(["a"])
        step.fit(df, SchemaInfo())
        assert step.schema_in().dtypes["a"] == "Int64"
        assert step.schema_out().dtypes["a"] == "Float64"

    def test_not_sparse(self) -> None:
        assert StepImputeMean(["a"]).produces_sparse is False
