"""End-to-end lifecycle tests for StepOther: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepOther + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, Recipe, SchemaInfo, StepOther, StepUnknown


def _data() -> dict[str, list[str | None]]:
    return {
        "color": ["red", "red", "red", "blue", "green"],
        "size": ["s", "s", "m", "l", "xl"],
    }


class TestEndToEnd:
    def test_pandas_pools_rare_levels(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOther(["color", "size"], threshold=0.4)])
        out = recipe.fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out["color"].tolist() == ["red", "red", "red", "other", "other"]
        assert out["size"].tolist() == ["s", "s", "other", "other", "other"]

    def test_polars_pools_rare_levels(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepOther(["color"], threshold=0.4))
        out = recipe.fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out["color"].to_list() == ["red", "red", "red", "other", "other"]

    def test_min_count_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepOther(["color"], min_count=3)]).fit(df).transform(df)
        assert out["color"].tolist() == ["red", "red", "red", "other", "other"]

    def test_default_threshold_through_recipe(self) -> None:
        # Neither criterion given -> the 0.05 default. Needs enough rows for 5%
        # to bite: "blue" is 1/21, below the default, while "red" is 20/21.
        df = pd.DataFrame({"color": ["red"] * 20 + ["blue"]})
        out = Recipe(steps=[StepOther(["color"])]).fit(df).transform(df)
        assert out["color"].tolist() == ["red"] * 20 + ["other"]

    def test_default_threshold_keeps_boundary_level(self) -> None:
        # The other side of the same boundary: 1/20 == 0.05 exactly, so "blue"
        # survives. Together these two pin the default to 0.05 through the step.
        df = pd.DataFrame({"color": ["red"] * 19 + ["blue"]})
        out = Recipe(steps=[StepOther(["color"])]).fit(df).transform(df)
        assert out["color"].tolist() == ["red"] * 19 + ["blue"]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe(steps=[StepOther(["color"], threshold=0.4)]).fit(df).transform(df)
        assert out["size"].tolist() == ["s", "s", "m", "l", "xl"]

    def test_custom_level_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepOther(["color"], threshold=0.4, other_level="rare")])
        out = recipe.fit(df).transform(df)
        assert out["color"].tolist() == ["red", "red", "red", "rare", "rare"]

    def test_unseen_level_pooled_on_new_data(self) -> None:
        train = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepOther(["color"], threshold=0.4)]).fit(train)
        new = pd.DataFrame({"color": ["red", "purple"], "size": ["s", "s"]})
        out = fitted.transform(new)
        assert out["color"].tolist() == ["red", "other"]

    def test_nulls_survive_transform(self) -> None:
        train = pd.DataFrame({"color": ["red", "red", "red", "blue", None]})
        fitted = Recipe(steps=[StepOther(["color"], threshold=0.4)]).fit(train)
        out = fitted.transform(train)
        assert out["color"].isna().tolist() == [False, False, False, False, True]

    def test_composes_with_step_unknown(self) -> None:
        # StepUnknown first gives missingness its own level; StepOther then sees
        # it as an ordinary level and pools it if it is rare.
        train = pd.DataFrame({"color": ["red", "red", "red", "blue", None]})
        recipe = (
            Recipe()
            .add(StepUnknown(["color"]))
            .add(StepOther(["color"], threshold=0.4))
        )
        out = recipe.fit(train).transform(train)
        assert out["color"].tolist() == ["red", "red", "red", "other", "other"]

    def test_categorical_input_pooled(self) -> None:
        df = pd.DataFrame({"c": pd.Categorical(["x", "x", "x", "y"])})
        out = Recipe(steps=[StepOther(["c"], threshold=0.4)]).fit(df).transform(df)
        assert out["c"].tolist() == ["x", "x", "x", "other"]

    def test_collision_warns_through_recipe(self, snapshot: SnapshotAssertion) -> None:
        df = pd.DataFrame({"c": ["a", "a", "other"]})
        with pytest.warns(UserWarning) as record:
            Recipe(steps=[StepOther(["c"], threshold=0.4)]).fit(df)
        assert str(record[0].message) == snapshot

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepOther(["color"], threshold=0.4)]).fit(df)
        assert all(step.trained for step in fitted.steps)


class TestStepOtherUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepOther(["c"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepOther(["c"]).id == "other"

    def test_explicit_id(self) -> None:
        assert StepOther(["c"], id="pool_rare").id == "pool_rare"

    def test_fit_returns_self(self) -> None:
        step = StepOther(["color"], threshold=0.4)
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            StepOther(["color"]).transform(nw.from_native(pd.DataFrame(_data())))
        assert str(excinfo.value) == snapshot

    def test_schema_in_before_fit_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(RuntimeError) as excinfo:
            StepOther(["color"]).schema_in()
        assert str(excinfo.value) == snapshot

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepOther(["color"]).schema_out()

    def test_accepts_categorical_dtypes(self) -> None:
        accepted = StepOther(["c"]).accepts_dtypes()
        assert nw.String in accepted
        assert nw.Categorical in accepted
        assert nw.Enum in accepted
        assert nw.Float64 not in accepted

    def test_schema_out_marks_recoded_columns_string(self) -> None:
        step = StepOther(["color", "size"], threshold=0.4)
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        so = step.schema_out()
        assert so.columns == ["color", "size"]
        assert so.dtypes == {"color": "String", "size": "String"}

    def test_not_sparse(self) -> None:
        assert StepOther(["c"]).produces_sparse is False
