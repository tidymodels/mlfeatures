"""End-to-end lifecycle tests for StepPls: Recipe -> fit -> transform.

Exercises the ``MANY_TO_MANY`` contract through the full vertical slice on both
pandas and polars: the selected numeric columns are dropped and replaced by
PLS component columns, each descending from the whole input set.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from mlfeatures import Mode, Recipe, StepNormalize, StepPls


def _data() -> dict[str, list[float]]:
    return {
        "id": [10.0, 20.0, 30.0, 40.0],
        "x": [1.0, 2.0, 3.0, 4.0],
        "y": [2.1, 3.9, 6.1, 8.0],
        "out": [1.0, 2.0, 2.5, 4.5],
    }


class TestEndToEnd:
    def test_pandas_replaces_inputs_with_components(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=2))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["id", "out", "PLS1", "PLS2"]
        assert out["id"].tolist() == [10.0, 20.0, 30.0, 40.0]

    def test_polars_replaces_inputs_with_components(self) -> None:
        df = pl.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=1))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["id", "out", "PLS1"]

    def test_num_comp_capped_at_input_count(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=5))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["id", "out", "PLS1", "PLS2"]

    def test_custom_prefix(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=2, prefix="comp"))
            .fit(df)
            .transform(df)
        )
        assert list(out.columns) == ["id", "out", "comp1", "comp2"]

    def test_components_attributed_to_input_set(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="out", num_comp=2)]).fit(df)
        parents = fitted.steps[0].output_parents()
        assert parents == {"PLS1": ["x", "y"], "PLS2": ["x", "y"]}

    def test_source_columns_lose_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="out", num_comp=2)]).fit(df)
        assert fitted.roles.of("x") == set()
        assert "feature" in fitted.roles.of("PLS1")

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=2, keep_original_cols=True))
            .fit(df)
            .transform(df)
        )
        assert set(out.columns) == {"id", "out", "x", "y", "PLS1", "PLS2"}
        assert out["x"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepPls(["x", "y"], target="out", num_comp=2, keep_original_cols=True)
            ]
        ).fit(df)
        assert "feature" in fitted.roles.of("x")
        assert "feature" in fitted.roles.of("PLS1")

    def test_keep_original_cols_leaves_produced_columns_alone(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepPls(["x", "y"], target="out", num_comp=2, keep_original_cols=True)
            ]
        ).fit(df)
        step = fitted.steps[0]
        assert step.schema_out().columns == ["PLS1", "PLS2"]
        assert step.output_parents() == {"PLS1": ["x", "y"], "PLS2": ["x", "y"]}

    def test_retained_sources_are_selectable_downstream(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="out", num_comp=2, keep_original_cols=True))
            .add(StepNormalize(["x"]))
            .fit(df)
            .transform(df)
        )
        assert out["x"].mean() == pytest.approx(0.0, abs=1e-12)

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="out", num_comp=2)]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_target_column_is_not_consumed(self) -> None:
        # The target survives untouched: it is read at fit time only, and is
        # not declared as an input feature.
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="out", num_comp=1)]).fit(df)
        assert "out" not in fitted.steps[0].schema_in().columns
        out = fitted.transform(df)
        assert out["out"].tolist() == [1.0, 2.0, 2.5, 4.5]


class TestCategoricalTargetEndToEnd:
    def _class_data(self) -> dict[str, list[float] | list[str]]:
        return {
            "id": [10.0, 20.0, 30.0, 40.0],
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 1.0, 10.0, 11.0],
            "label": ["a", "a", "b", "b"],
        }

    def test_pandas_replaces_inputs_with_components(self) -> None:
        df = pd.DataFrame(self._class_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="label", num_comp=1))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["id", "label", "PLS1"]

    def test_polars_replaces_inputs_with_components(self) -> None:
        df = pl.DataFrame(self._class_data())
        out = (
            Recipe()
            .add(StepPls(["x", "y"], target="label", num_comp=1))
            .fit(df)
            .transform(df)
        )
        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["id", "label", "PLS1"]

    def test_target_column_is_not_consumed(self) -> None:
        df = pd.DataFrame(self._class_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="label", num_comp=1)]).fit(df)
        assert "label" not in fitted.steps[0].schema_in().columns
        out = fitted.transform(df)
        assert out["label"].tolist() == ["a", "a", "b", "b"]


class TestStepPls:
    def test_mode_is_many_to_many(self) -> None:
        assert StepPls(["x", "y"], target="out").mode is Mode.MANY_TO_MANY

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepPls(["x", "y"], target="out", num_comp=2)]).fit(df)
        assert all(step.trained for step in fitted.steps)
