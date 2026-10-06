"""End-to-end lifecycle tests for StepGeoDistance: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepGeoDistance + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic. Structured to parallel
``tests/test_step_class_distance_e2e.py``.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Mode, Recipe, SchemaInfo, StepGeoDistance

UNION_STATION = (38.8986312, -77.0062457)


def _data() -> dict[str, list[float]]:
    return {
        "lat": [38.8986312, 38.888504],
        "lon": [-77.0062457, -77.026352],
        "passthrough": [9.0, 9.0],
    }


# ---------------------------------------------------------------------------
# Recipe-level end-to-end behaviour
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pandas_replaces_lat_lon(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[
                StepGeoDistance(
                    "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
                )
            ]
        )
        out = recipe.fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert set(out.columns) == {"passthrough", "geo_dist"}

    def test_polars_produces_distance_column(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(
            StepGeoDistance(
                "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
            )
        )
        out = recipe.fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert "geo_dist" in out.columns

    def test_row_at_reference_has_zero_distance(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(
            steps=[
                StepGeoDistance(
                    "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
                )
            ]
        )
        out = recipe.fit(df).transform(df)
        assert out["geo_dist"].iloc[0] == pytest.approx(0.0, abs=1e-6)

    def test_transform_uses_fit_time_reference_not_new_data(self) -> None:
        train = pd.DataFrame(_data())
        new = pd.DataFrame({"lat": [0.0], "lon": [0.0], "passthrough": [1.0]})
        fitted = Recipe(
            steps=[
                StepGeoDistance(
                    "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
                )
            ]
        ).fit(train)
        out = fitted.transform(new)
        # Reference point is fixed at construction time, not re-derived from
        # ``new`` — distance from (0, 0) to Union Station is large but finite.
        assert out["geo_dist"].iloc[0] > 0.0

    def test_source_columns_lose_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepGeoDistance(
                    "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
                )
            ]
        ).fit(df)
        assert fitted.roles.of("lat") == set()
        assert fitted.roles.of("lon") == set()
        assert "feature" in fitted.roles.of("geo_dist")

    def test_keep_original_cols_retains_sources(self) -> None:
        df = pd.DataFrame(_data())
        out = (
            Recipe()
            .add(
                StepGeoDistance(
                    "lat",
                    "lon",
                    ref_lat=UNION_STATION[0],
                    ref_lon=UNION_STATION[1],
                    keep_original_cols=True,
                )
            )
            .fit(df)
            .transform(df)
        )
        assert set(out.columns) == {"lat", "lon", "passthrough", "geo_dist"}

    def test_keep_original_cols_retains_source_roles(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(
            steps=[
                StepGeoDistance(
                    "lat",
                    "lon",
                    ref_lat=UNION_STATION[0],
                    ref_lon=UNION_STATION[1],
                    keep_original_cols=True,
                )
            ]
        ).fit(df)
        assert "feature" in fitted.roles.of("lat")
        assert "feature" in fitted.roles.of("geo_dist")

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(
            steps=[
                StepGeoDistance(
                    "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
                )
            ]
        ).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


# ---------------------------------------------------------------------------
# StepGeoDistance — unit-level invariants
# ---------------------------------------------------------------------------


class TestStepGeoDistanceUnit:
    def test_mode_is_many_to_many(self) -> None:
        step = StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0)
        assert step.mode is Mode.MANY_TO_MANY

    def test_default_id(self) -> None:
        assert (
            StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0).id == "geo_distance"
        )

    def test_explicit_id(self) -> None:
        step = StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0, id="my_dist")
        assert step.id == "my_dist"

    def test_keep_original_cols_default_false(self) -> None:
        step = StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0)
        assert step.keep_original_cols is False

    def test_fit_returns_self(self) -> None:
        step = StepGeoDistance(
            "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
        )
        fitted = step.fit(_nw_frame(), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0)
        with pytest.raises(RuntimeError):
            step.transform(_nw_frame())

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0).schema_out()

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepGeoDistance(
            "lat", "lon", ref_lat=0.0, ref_lon=0.0
        ).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_is_single_named_column(self) -> None:
        step = StepGeoDistance(
            "lat",
            "lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
            name="dist",
        )
        step.fit(_nw_frame(), SchemaInfo())
        assert step.schema_out().columns == ["dist"]
        assert step.schema_out().dtypes == {"dist": "Float64"}

    def test_output_parents(self) -> None:
        step = StepGeoDistance(
            "lat", "lon", ref_lat=UNION_STATION[0], ref_lon=UNION_STATION[1]
        )
        step.fit(_nw_frame(), SchemaInfo())
        assert step.output_parents() == {"geo_dist": ["lat", "lon"]}

    def test_not_sparse(self) -> None:
        assert (
            StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=0.0).produces_sparse
            is False
        )

    def test_invalid_ref_lat_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepGeoDistance("lat", "lon", ref_lat=91.0, ref_lon=0.0)
        assert str(excinfo.value) == snapshot

    def test_invalid_ref_lon_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepGeoDistance("lat", "lon", ref_lat=0.0, ref_lon=181.0)
        assert str(excinfo.value) == snapshot

    def test_euclidean_metric_does_not_validate_range(self) -> None:
        # Only the haversine metric requires geographic coordinate ranges.
        step = StepGeoDistance(
            "lat", "lon", ref_lat=1000.0, ref_lon=1000.0, metric="euclidean"
        )
        assert step.ref_lat == 1000.0


def _nw_frame() -> nw.DataFrame[pd.DataFrame]:
    return nw.from_native(pd.DataFrame(_data()))
