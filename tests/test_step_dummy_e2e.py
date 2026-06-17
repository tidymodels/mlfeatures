"""End-to-end lifecycle tests for StepDummy: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDummy + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepDummy


def _data() -> dict[str, list[str]]:
    # sorted levels: blue, green, red -> reference (dropped) level is "blue".
    return {
        "color": ["red", "green", "blue", "red"],
        "other": ["x", "y", "x", "y"],
    }


class TestEndToEnd:
    def test_pandas_expands_to_reference_coded_indicators(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDummy(["color"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # reference level "blue" dropped; source column dropped by default.
        assert "color" not in out.columns
        assert "color_green" in out.columns
        assert "color_red" in out.columns
        assert "color_blue" not in out.columns
        assert out["color_red"].tolist() == [1, 0, 0, 1]
        assert out["color_green"].tolist() == [0, 1, 0, 0]

    def test_polars_expands_to_reference_coded_indicators(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepDummy(["color"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert "color" not in out.columns
        assert out["color_red"].to_list() == [1, 0, 0, 1]

    def test_keep_original_cols(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDummy(["color"], keep_original_cols=True)])
        out = recipe.fit(df).transform(df)
        assert "color" in out.columns
        assert "color_red" in out.columns

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDummy(["color"])])
        out = recipe.fit(df).transform(df)
        assert out["other"].tolist() == ["x", "y", "x", "y"]

    def test_transform_uses_train_levels_not_new_data(self) -> None:
        train = pd.DataFrame({"g": ["a", "b", "c"]})
        fitted = Recipe(steps=[StepDummy(["g"])]).fit(train)
        new = pd.DataFrame({"g": ["a", "b"]})
        out = fitted.transform(new)
        # columns are fixed by training levels (reference "a" dropped): g_b, g_c.
        assert "g_c" in out.columns
        assert out["g_b"].tolist() == [0, 1]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDummy(["color"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepDummy(["color"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestNameCollisions:
    """Generated indicator names are validated centrally by the recipe.

    ``StepDummy`` no longer checks names itself; the recipe rejects collisions
    via ``Schema.check_additions`` before the step runs, on every backend.
    """

    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pd", "pl"])
    def test_generated_name_collides_with_existing_column_raises(
        self, ctor: type
    ) -> None:
        # "color" expands to color_green/color_red; color_red already exists.
        df = ctor({"color": ["red", "green", "blue", "red"], "color_red": [1, 2, 3, 4]})
        recipe = Recipe(steps=[StepDummy(["color"], keep_original_cols=True)])
        with pytest.raises(ValueError, match="already exist"):
            recipe.fit(df)

    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pd", "pl"])
    def test_two_generated_columns_with_same_name_raises(self, ctor: type) -> None:
        # Default naming is f"{col}_{level}". Column "a" retains level "b_c"
        # (reference "a1" dropped) -> "a_b_c"; column "a_b" retains level "c"
        # (reference "a0" dropped) -> "a_b_c". Both indicators want one name.
        # This intra-step clash used to slip past the centralized check (pandas
        # silently kept one column); check_additions now rejects it up front.
        df = ctor({"a": ["a1", "b_c", "a1", "b_c"], "a_b": ["a0", "c", "a0", "c"]})
        recipe = Recipe(steps=[StepDummy(["a", "a_b"])])
        with pytest.raises(ValueError, match="duplicate column name"):
            recipe.fit(df)


class TestStepDummyUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepDummy(["color"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepDummy(["color"]).id == "dummy"

    def test_explicit_id(self) -> None:
        assert StepDummy(["color"], id="my_dummy").id == "my_dummy"

    def test_fit_returns_self(self) -> None:
        step = StepDummy(["color"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepDummy(["color"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_not_sparse(self) -> None:
        assert StepDummy(["color"]).produces_sparse is False
