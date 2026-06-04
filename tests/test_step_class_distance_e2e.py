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

from shoez import Mode, Recipe, SchemaInfo, StepClassDistance


def _data() -> dict[str, list[float] | list[str]]:
    """Two-predictor, two-class training data with one passthrough column."""
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
    def test_pandas_appends_distance_columns(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[
                StepClassDistance(["x", "y"], outcome="label", log_distance=False),
            ]
        )
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # Original four columns are preserved in order; two new ones appended.
        assert list(out.columns) == [
            "x",
            "y",
            "passthrough",
            "label",
            "distance_a",
            "distance_b",
        ]

    def test_polars_appends_distance_columns(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(
            StepClassDistance(["x", "y"], outcome="label", log_distance=False)
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
                StepClassDistance(["x", "y"], outcome="label", log_distance=False),
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
                # Same predictor values, completely different labels — if the
                # centroids were re-estimated from ``new``, the values would change.
                "x": [0.0, 2.0, 10.0, 12.0],
                "y": [0.0, 4.0, 10.0, 14.0],
                "label": ["b", "b", "a", "a"],
            }
        )
        fitted = Recipe(
            steps=[
                StepClassDistance(["x", "y"], outcome="label", log_distance=False),
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
            steps=[StepClassDistance(["x", "y"], outcome="label")],
        )
        out = recipe.fit(df).transform(df)
        # Every distance is < raw distance (since log compresses for d > 1)
        # but more importantly all finite — even when a row sits on a centroid.
        for col in ("distance_a", "distance_b"):
            assert out[col].notna().all()

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepClassDistance(["x", "y"], outcome="label")]).fit(
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
            steps=[StepClassDistance(["x", "y"], outcome="label", log_distance=False)]
        )
        out = recipe.fit(df).transform(df)
        # Original rows preserved (no NaN introduced) and indices kept.
        assert out["x"].notna().all()
        assert out["distance_a"].notna().all()
        assert out["distance_b"].notna().all()
        assert list(out.index) == [100, 200, 300, 400, 500, 600]


# ---------------------------------------------------------------------------
# StepClassDistance — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepClassDistanceUnit:
    def test_mode_is_adding(self) -> None:
        step = StepClassDistance(["x", "y"], outcome="label")
        assert step.mode is Mode.ADDING

    def test_default_id(self) -> None:
        assert StepClassDistance(["x"], outcome="label").id == "class_distance"

    def test_explicit_id(self) -> None:
        step = StepClassDistance(["x"], outcome="label", id="my_dist")
        assert step.id == "my_dist"

    def test_fit_returns_self(self) -> None:
        step = StepClassDistance(["x", "y"], outcome="label")
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepClassDistance(["x", "y"], outcome="label")
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepClassDistance(["x"], outcome="label").schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepClassDistance(["x"], outcome="label").schema_out()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepClassDistance(["x"], outcome="label").accepts_dtypes()
        # Same numeric set as StepNormalize.
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_in_lists_predictor_columns_only(self) -> None:
        step = StepClassDistance(["x", "y"], outcome="label")
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        # The outcome column is consumed at fit time but not declared an input.
        assert step.schema_in().columns == ["x", "y"]

    def test_schema_out_lists_new_distance_columns(self) -> None:
        step = StepClassDistance(["x", "y"], outcome="label")
        step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        # Two classes -> two distance columns, both Float64.
        assert step.schema_out().columns == ["distance_a", "distance_b"]
        assert step.schema_out().dtypes == {
            "distance_a": "Float64",
            "distance_b": "Float64",
        }

    def test_not_sparse(self) -> None:
        assert StepClassDistance(["x"], outcome="label").produces_sparse is False
