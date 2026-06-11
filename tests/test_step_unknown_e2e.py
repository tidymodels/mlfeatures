"""End-to-end lifecycle tests for StepUnknown: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepUnknown + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepUnknown


def _data() -> dict[str, list[str | None]]:
    return {
        "color": ["red", "blue", None, "red"],
        "size": ["s", None, "m", "l"],
    }


class TestEndToEnd:
    def test_pandas_recodes_nulls(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepUnknown(["color", "size"])]).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out["color"].tolist() == ["red", "blue", "unknown", "red"]
        assert out["size"].tolist() == ["s", "unknown", "m", "l"]

    def test_polars_recodes_nulls(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepUnknown(["color", "size"])).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out["color"].to_list() == ["red", "blue", "unknown", "red"]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepUnknown(["color"])]).fit(df).transform(df)
        # "size" not selected -> its null is left as a null (NaN/None).
        assert out["color"].tolist() == ["red", "blue", "unknown", "red"]
        assert out["size"].isna().tolist() == [False, True, False, False]

    def test_custom_level_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepUnknown(["color"], new_level="MISSING")])
        out = recipe.fit(df).transform(df)
        assert out["color"].tolist() == ["red", "blue", "MISSING", "red"]

    def test_transform_uses_fit_columns_on_new_data(self) -> None:
        train = pd.DataFrame({"color": ["red", "blue"]})  # no nulls at fit
        fitted = Recipe(steps=[StepUnknown(["color"])]).fit(train)
        new = pd.DataFrame({"color": ["red", None]})  # null appears at transform
        out = fitted.transform(new)
        assert out["color"].tolist() == ["red", "unknown"]

    def test_categorical_input_recoded(self) -> None:
        df = pd.DataFrame({"c": pd.Categorical(["x", None, "x"])})
        out = Recipe(steps=[StepUnknown(["c"])]).fit(df).transform(df)
        assert out["c"].tolist() == ["x", "unknown", "x"]

    def test_collision_warns_through_recipe(self) -> None:
        df = pd.DataFrame({"c": ["a", "unknown", None]})
        with pytest.warns(UserWarning, match="already exists"):
            Recipe(steps=[StepUnknown(["c"])]).fit(df)

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepUnknown(["color"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepUnknownUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepUnknown(["c"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepUnknown(["c"]).id == "unknown"

    def test_explicit_id(self) -> None:
        assert StepUnknown(["c"], id="fill_na").id == "fill_na"

    def test_fit_returns_self(self) -> None:
        step = StepUnknown(["color"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepUnknown(["color"]).transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepUnknown(["color"]).schema_in()

    def test_accepts_categorical_dtypes(self) -> None:
        accepted = StepUnknown(["c"]).accepts_dtypes()
        assert nw.String in accepted
        assert nw.Categorical in accepted
        assert nw.Enum in accepted
        assert nw.Float64 not in accepted

    def test_schema_out_marks_recoded_columns_string(self) -> None:
        step = StepUnknown(["color", "size"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert step.schema_out().dtypes["color"] == "String"

    def test_schema_out_preserves_multi_column_order(self) -> None:
        step = StepUnknown(["color", "size"])
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        so = step.schema_out()
        assert so.columns == ["color", "size"]
        assert so.dtypes == {"color": "String", "size": "String"}

    def test_schema_out_recodes_every_selected_column_to_string(self) -> None:
        # The resolved selection (_schema_in.columns) is the recoded set: every
        # selected column becomes String regardless of its input dtype.
        from shoez.types import ColSchema

        step = StepUnknown(["color", "n"])
        step._schema_in = ColSchema(
            columns=["color", "n"], dtypes={"color": "String", "n": "Int64"}
        )
        so = step.schema_out()
        assert so.columns == ["color", "n"]
        assert so.dtypes == {"color": "String", "n": "String"}

    def test_not_sparse(self) -> None:
        assert StepUnknown(["c"]).produces_sparse is False
