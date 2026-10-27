"""End-to-end lifecycle tests for StepClassDistance: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepClassDistance + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Structured to parallel ``tests/test_step_normalize_e2e.py``.
"""

from __future__ import annotations

import math

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from mlfeatures import Mode, Recipe, SchemaInfo, StepClassDistance


def _data() -> dict[str, list[float] | list[str]]:
    """Two-feature, two-class training data with one passthrough column."""
    return {
        "x": [0.0, 2.0, 10.0, 12.0],
        "y": [0.0, 4.0, 10.0, 14.0],
        "passthrough": [9.0, 9.0, 9.0, 9.0],
        "label": ["a", "a", "b", "b"],
    }


# ---------------------------------------------------------------------------
# Recipe-level end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pandas_replaces_selected_columns(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[
                StepClassDistance(["x", "y"], target="label", log_distance=False),
            ]
        )
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # The selected columns are gone; the unselected ones pass through and
        # the two distance columns take the selection's place.
        assert set(out.columns) == {
            "passthrough",
            "label",
            "distance_a",
            "distance_b",
        }

    def test_polars_produces_distance_columns(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(
            StepClassDistance(["x", "y"], target="label", log_distance=False)
        )
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns)[-2:] == ["distance_a", "distance_b"]

    def test_distance_values_match_hand_calc(self) -> None:
        # Hand calc: class "a" centroid (1, 2), class "b" centroid (11, 12).
        # Row 0 = (0, 0): distance to a = sqrt(5); distance to b = sqrt(265).
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[
                StepClassDistance(["x", "y"], target="label", log_distance=False),
            ]
        )
        out = recipe.fit(df).transform(df)
        assert out["distance_a"].iloc[0] == pytest.approx(math.sqrt(5.0))
        assert out["distance_b"].iloc[0] == pytest.approx(math.sqrt(265.0))

    def test_transform_uses_train_centroids_not_new_data(self) -> None:
        # If transform re-estimated, the centroids would shift to match the
        # new data; instead they must stay pinned to the training set.
        train = pd.DataFrame(
            {
                "x": [0.0, 2.0, 10.0, 12.0],
                "y": [0.0, 4.0, 10.0, 14.0],
                "label": ["a", "a", "b", "b"],
            }
        )
        new = pd.DataFrame(
            {
                # Same feature values, completely different labels. If the
                # centroids were re-estimated from ``new``, the values would change.
                "x": [0.0, 2.0, 10.0, 12.0],
                "y": [0.0, 4.0, 10.0, 14.0],
                "label": ["b", "b", "a", "a"],
            }
        )
        fitted = Recipe(
            steps=[
                StepClassDistance(["x", "y"], target="label", log_distance=False),
            ]
        ).fit(train)
        out = fitted.transform(new)
        # Distances computed against the *training* centroids (a=(1,2), b=(11,12)),
        # so row 0 of ``new`` (also (0, 0)) still gives sqrt(5) for distance_a.
        assert out["distance_a"].iloc[0] == pytest.approx(math.sqrt(5.0))

    def test_log_transform_default(self) -> None:
        # Default is log_distance=True
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[StepClassDistance(["x", "y"], target="label")],
        )
        out = recipe.fit(df).transform(df)
        # Every distance is < raw distance (since log compresses for d > 1)
        # but more importantly all finite, even when a row sits on a centroid.
        for col in ("distance_a", "distance_b"):
            assert out[col].notna().all()

    def test_source_columns_lose_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepClassDistance(["x", "y"], target="label")]).fit(df)
        assert fitted.roles.of("x") == set()
        assert fitted.roles.of("y") == set()
        # The target column is read but not consumed, so it keeps its role.
        assert fitted.roles.of("label") != set()
        assert "feature" in fitted.roles.of("distance_a")

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(
                StepClassDistance(
                    ["x", "y"],
                    target="label",
                    log_distance=False,
                    keep_original_cols=True,
                )
            )
            .fit(df)
            .transform(df)
        )
        assert set(out.columns) == {
            "x",
            "y",
            "passthrough",
            "label",
            "distance_a",
            "distance_b",
        }
        assert out["x"].tolist() == [0.0, 2.0, 10.0, 12.0]

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepClassDistance(["x", "y"], target="label", keep_original_cols=True)
            ]
        ).fit(df)
        assert "feature" in fitted.roles.of("x")
        assert "feature" in fitted.roles.of("distance_a")

    def test_keep_original_cols_leaves_produced_columns_alone(self) -> None:
        # The flag governs the source columns only; what the step *produces* is
        # unchanged.
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepClassDistance(["x", "y"], target="label", keep_original_cols=True)
            ]
        ).fit(df)
        step = fitted.steps[0]
        assert step.schema_out().columns == ["distance_a", "distance_b"]
        assert step.output_parents() == {
            "distance_a": ["x", "y"],
            "distance_b": ["x", "y"],
        }

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepClassDistance(["x", "y"], target="label")]).fit(
            pd.DataFrame(_data())
        )
        assert all(step.trained for step in fitted.steps)

    def test_non_default_pandas_index_does_not_introduce_nan(self) -> None:
        # Regression: an earlier version used `nw.concat(how="horizontal")` to
        # append the distance columns onto ``data``. Pandas horizontal concat
        # aligns by *index*, so a frame with a shuffled / non-default index
        # (e.g. after sklearn.train_test_split) ended up with all-NaN rows
        # wherever the source index didn't match the distance frame's
        # RangeIndex. Fixed by switching to `with_columns`, which is positional.
        df = pd.DataFrame(
            {
                "x": [0.0, 2.0, 10.0, 12.0, 1.0, 3.0],
                "y": [0.0, 4.0, 10.0, 14.0, 1.0, 3.0],
                "label": ["a", "a", "b", "b", "a", "b"],
            },
            index=[100, 200, 300, 400, 500, 600],
        )
        recipe = Recipe(
            steps=[StepClassDistance(["x", "y"], target="label", log_distance=False)]
        )
        out = recipe.fit(df).transform(df)
        # Rows preserved (no NaN introduced) and indices kept.
        assert out["distance_a"].notna().all()
        assert out["distance_b"].notna().all()
        assert list(out.index) == [100, 200, 300, 400, 500, 600]


# ---------------------------------------------------------------------------
# StepClassDistance: unit-level invariants
# ---------------------------------------------------------------------------


class TestStepClassDistanceUnit:
    def test_mode_is_many_to_many(self) -> None:
        step = StepClassDistance(["x", "y"], target="label")
        assert step.mode is Mode.MANY_TO_MANY

    def test_default_id(self) -> None:
        assert StepClassDistance(["x"], target="label").id == "class_distance"

    def test_explicit_id(self) -> None:
        step = StepClassDistance(["x"], target="label", id="my_dist")
        assert step.id == "my_dist"

    def test_keep_original_cols_default_false(self) -> None:
        assert StepClassDistance(["x"], target="label").keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        step = StepClassDistance(["x"], target="label", keep_original_cols=True)
        assert step.keep_original_cols is True

    def test_fit_returns_self(self) -> None:
        step = StepClassDistance(["x", "y"], target="label")
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepClassDistance(["x", "y"], target="label")
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepClassDistance(["x"], target="label").schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepClassDistance(["x"], target="label").schema_out()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepClassDistance(["x"], target="label").accepts_dtypes()
        # Same numeric set as StepNormalize.
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_in_lists_feature_columns_only(self) -> None:
        step = StepClassDistance(["x", "y"], target="label")
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        # The target column is consumed at fit time but not declared an input.
        assert step.schema_in().columns == ["x", "y"]

    def test_schema_out_lists_new_distance_columns(self) -> None:
        step = StepClassDistance(["x", "y"], target="label")
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        # Two classes -> two distance columns, both Float64.
        assert step.schema_out().columns == ["distance_a", "distance_b"]
        assert step.schema_out().dtypes == {
            "distance_a": "Float64",
            "distance_b": "Float64",
        }

    def test_schema_out_sorts_classes_regardless_of_input_order(self) -> None:
        # Target labels appear in reverse-sorted order ("b" before "a"); the
        # distance columns must still come out sorted at the wrapper layer.
        df = pd.DataFrame(
            {
                "x": [10.0, 12.0, 0.0, 2.0],
                "y": [10.0, 14.0, 0.0, 4.0],
                "label": ["b", "b", "a", "a"],
            }
        )
        step = StepClassDistance(["x", "y"], target="label")
        step.fit(nw.from_native(df), SchemaInfo())
        assert step.schema_out().columns == ["distance_a", "distance_b"]

    def test_single_class_yields_single_distance_column(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0], "label": ["a", "a"]})
        step = StepClassDistance(["x", "y"], target="label")
        step.fit(nw.from_native(df), SchemaInfo())
        assert step.schema_out().columns == ["distance_a"]

    def test_missing_target_column_raises(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})  # no "label"
        step = StepClassDistance(["x", "y"], target="label")
        with pytest.raises(KeyError):
            step.fit(nw.from_native(df), SchemaInfo())

    def test_not_sparse(self) -> None:
        assert StepClassDistance(["x"], target="label").produces_sparse is False
