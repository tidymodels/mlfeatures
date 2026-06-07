"""End-to-end lifecycle tests for StepImputeMode: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepImputeMode + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepImputeMode


def _data() -> dict[str, list[object]]:
    return {
        "color": ["red", "red", "blue", None],  # mode = "red"
        "size": [1, 1, 2, 3],
    }


class TestEndToEnd:
    def test_pandas_imputes_categorical_mode(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputeMode(["color"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out["color"].tolist() == ["red", "red", "blue", "red"]

    def test_polars_imputes_categorical_mode(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepImputeMode(["color"])).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out["color"].to_list() == ["red", "red", "blue", "red"]

    def test_dtype_preserved_string(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputeMode(["color"])]).fit(df).transform(df)
        # mode imputation does not change dtype (unlike mean/median -> Float64):
        # the column stays a string, not promoted to a numeric type.
        assert nw.from_native(out)["color"].dtype == nw.String

    def test_transform_uses_train_mode_not_new_data(self) -> None:
        train = pd.DataFrame({"g": ["a", "a", "b"]})  # train mode = "a"
        fitted = Recipe(steps=[StepImputeMode(["g"])]).fit(train)
        new = pd.DataFrame({"g": ["b", "b", None]})  # new-data mode would be "b"
        out = fitted.transform(new)
        assert out["g"].tolist() == ["b", "b", "a"]  # filled with TRAIN mode

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepImputeMode(["color"])]).fit(df).transform(df)
        assert out["size"].tolist() == [1, 1, 2, 3]

    def test_all_null_column_warns_and_passes_through(self) -> None:
        df = pd.DataFrame(
            {
                "g": ["a", "a", None],
                "empty": pd.Series([None, None, None], dtype="object"),
            }
        )
        recipe = Recipe(steps=[StepImputeMode(["g", "empty"])])
        with pytest.warns(UserWarning, match="entirely null"):
            fitted = recipe.fit(df)
        out = fitted.transform(df)
        assert out["g"].tolist() == ["a", "a", "a"]
        assert out["empty"].isna().all()

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepImputeMode(["color"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepImputeMode(["color"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepImputeModeUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepImputeMode(["g"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepImputeMode(["g"]).id == "impute_mode"

    def test_explicit_id(self) -> None:
        assert StepImputeMode(["g"], id="impute_mode").id == "impute_mode"

    def test_fit_returns_self(self) -> None:
        step = StepImputeMode(["color"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputeMode(["color"]).transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepImputeMode(["color"]).schema_in()

    def test_accepts_categorical_dtypes_only(self) -> None:
        accepted = StepImputeMode(["g"]).accepts_dtypes()
        # mode imputation is categorical-only
        assert nw.String in accepted
        assert nw.Categorical in accepted
        assert nw.Enum in accepted
        # numeric dtypes are rejected (use mean/median imputation instead)
        assert nw.Float64 not in accepted
        assert nw.Int64 not in accepted

    def test_schema_out_preserves_dtype(self) -> None:
        step = StepImputeMode(["color"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        # dtype unchanged from schema_in
        assert step.schema_out().dtypes["color"] == step.schema_in().dtypes["color"]

    def test_not_sparse(self) -> None:
        assert StepImputeMode(["g"]).produces_sparse is False
