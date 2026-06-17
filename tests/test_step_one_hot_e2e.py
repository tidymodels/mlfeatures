"""End-to-end lifecycle tests for StepOneHot: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepOneHot + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from shoez import Mode, Recipe, SchemaInfo, StepOneHot


def _data() -> dict[str, list[str]]:
    return {
        "color": ["red", "green", "blue", "red"],
        "other": ["x", "y", "x", "y"],
    }


class TestEndToEnd:
    def test_pandas_retains_all_levels(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOneHot(["color"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # one-hot retains every level (k columns); source column dropped.
        assert "color" not in out.columns
        assert "color_blue" in out.columns
        assert "color_green" in out.columns
        assert "color_red" in out.columns
        assert out["color_red"].tolist() == [1, 0, 0, 1]
        assert out["color_blue"].tolist() == [0, 0, 1, 0]

    def test_polars_retains_all_levels(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepOneHot(["color"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert "color_blue" in out.columns
        assert out["color_red"].to_list() == [1, 0, 0, 1]

    def test_keep_original_cols(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOneHot(["color"], keep_original_cols=True)])
        out = recipe.fit(df).transform(df)
        assert "color" in out.columns
        assert "color_red" in out.columns

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOneHot(["color"])])
        out = recipe.fit(df).transform(df)
        assert out["other"].tolist() == ["x", "y", "x", "y"]

    def test_transform_uses_train_levels_not_new_data(self) -> None:
        train = pd.DataFrame({"g": ["a", "b", "c"]})
        fitted = Recipe(steps=[StepOneHot(["g"])]).fit(train)
        new = pd.DataFrame({"g": ["a", "b"]})
        out = fitted.transform(new)
        # columns fixed by training levels: g_a, g_b, g_c (all retained).
        assert {"g_a", "g_b", "g_c"} <= set(out.columns)
        assert out["g_a"].tolist() == [1, 0]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepOneHot(["color"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepOneHot(["color"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestNameCollisions:
    """Generated indicator names are validated centrally by the recipe."""

    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pd", "pl"])
    def test_generated_name_collides_with_existing_column_raises(
        self, ctor: type
    ) -> None:
        df = ctor({"color": ["red", "green", "red"], "color_red": [1, 2, 3]})
        recipe = Recipe(steps=[StepOneHot(["color"], keep_original_cols=True)])
        with pytest.raises(ValueError, match="already exist"):
            recipe.fit(df)

    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame], ids=["pd", "pl"])
    def test_two_generated_columns_with_same_name_raises(self, ctor: type) -> None:
        # One-hot keeps every level, so "a" + level "b_c" and "a_b" + level "c"
        # both want the name "a_b_c".
        df = ctor({"a": ["b_c", "z", "b_c"], "a_b": ["c", "z", "c"]})
        recipe = Recipe(steps=[StepOneHot(["a", "a_b"])])
        with pytest.raises(ValueError, match="duplicate column name"):
            recipe.fit(df)


class TestStepOneHotUnit:
    def test_mode_is_one_to_many(self) -> None:
        assert StepOneHot(["color"]).mode is Mode.ONE_TO_MANY

    def test_default_id(self) -> None:
        assert StepOneHot(["color"]).id == "one_hot"

    def test_explicit_id(self) -> None:
        assert StepOneHot(["color"], id="my_oh").id == "my_oh"

    def test_fit_returns_self(self) -> None:
        step = StepOneHot(["color"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepOneHot(["color"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_not_sparse(self) -> None:
        assert StepOneHot(["color"]).produces_sparse is False
