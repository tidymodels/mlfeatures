"""End-to-end lifecycle tests for StepPca: Recipe -> fit -> transform.

Exercises the ``MANY_TO_MANY`` contract through the full vertical slice on both
pandas and polars: the selected numeric columns are dropped and replaced by
principal-component columns, each descending from the whole input set.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from shoez import Mode, Recipe, StepPca


def _data() -> dict[str, list[float]]:
    return {
        "id": [10.0, 20.0, 30.0, 40.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.1, 3.9, 6.1, 8.0],
    }


class TestEndToEnd:
    def test_pandas_replaces_inputs_with_components(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepPca(["x", "y"], num_comp=2)).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["id", "PC1", "PC2"]
        assert out["id"].tolist() == [10.0, 20.0, 30.0, 40.0]

    def test_polars_replaces_inputs_with_components(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepPca(["x", "y"], num_comp=1)).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["id", "PC1"]

    def test_num_comp_capped_at_input_count(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepPca(["x", "y"], num_comp=5)).fit(df).transform(df)
        assert list(out.columns) == ["id", "PC1", "PC2"]

    def test_custom_prefix(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPca(["x", "y"], num_comp=2, prefix="comp"))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["id", "comp1", "comp2"]

    def test_components_attributed_to_input_set(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPca(["x", "y"], num_comp=2)]).fit(df)
        parents = fitted.steps[0].output_parents()
        assert parents == {"PC1": ["x", "y"], "PC2": ["x", "y"]}

    def test_source_columns_lose_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPca(["x", "y"], num_comp=2)]).fit(df)
        assert fitted.roles.of("x") == set()
        assert "predictor" in fitted.roles.of("PC1")

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPca(["x", "y"], num_comp=2)]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))


class TestStepPca:
    def test_mode_is_many_to_many(self) -> None:
        assert StepPca(["x", "y"]).mode is Mode.MANY_TO_MANY

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPca(["x", "y"], num_comp=2)]).fit(df)
        assert all(step.trained for step in fitted.steps)
