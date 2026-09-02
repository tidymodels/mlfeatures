"""Tests for fit_geo_distance and apply_geo_distance.

Mirrors the structure of ``tests/test_step_class_distance.py``: pytest with
class-based grouping, a tiny ``_df`` helper, narwhals frames built from
pandas. The functions under test operate on the internal narwhals
:data:`~shoez.types.Frame` type; the public contract is narwhals-in,
narwhals-out and is backend-agnostic by construction, so we don't
parametrise pandas vs polars at this layer (the end-to-end test does that
for the ``StepGeoDistance`` wrapper).
"""

from __future__ import annotations

import math
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from shoez.step_geo_distance import GeoDistanceFit, apply_geo_distance, fit_geo_distance


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    """Build a narwhals frame from a plain dict (pandas backend)."""
    return nw.from_native(pd.DataFrame(data), eager_only=True)


# Union Station, Washington D.C. and the Smithsonian Castle, ~1.9km apart.
UNION_STATION = (38.8986312, -77.0062457)
COORDS: dict[str, list[float]] = {
    "lat": [38.8986312, 38.888504],
    "lon": [-77.0062457, -77.026352],
}


# ---------------------------------------------------------------------------
# fit_geo_distance — return type & fields
# ---------------------------------------------------------------------------


class TestFitReturnType:
    def test_returns_geo_distance_fit(self) -> None:
        result = fit_geo_distance(
            _df(COORDS),
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        assert isinstance(result, GeoDistanceFit)

    def test_fields_are_stored_verbatim(self) -> None:
        result = fit_geo_distance(
            _df(COORDS),
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
            metric="euclidean",
            log_distance=True,
            name="dist_to_station",
        )
        assert result.lat == "lat"
        assert result.lon == "lon"
        assert result.ref_lat == UNION_STATION[0]
        assert result.ref_lon == UNION_STATION[1]
        assert result.metric == "euclidean"
        assert result.log_distance is True
        assert result.name == "dist_to_station"

    def test_defaults(self) -> None:
        result = fit_geo_distance(
            _df(COORDS),
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        assert result.metric == "haversine"
        assert result.log_distance is False
        assert result.name == "geo_dist"


# ---------------------------------------------------------------------------
# apply_geo_distance — output shape & passthrough
# ---------------------------------------------------------------------------


class TestApplyShape:
    def test_replaces_lat_lon_with_single_column(self) -> None:
        selected = _df(COORDS)
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit)
        assert out.columns == ["geo_dist"]

    def test_unselected_columns_pass_through_unchanged(self) -> None:
        data = _df({**COORDS, "z": [7.0, 8.0]})
        fit = fit_geo_distance(
            data,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(data, fit)
        assert "z" in out.columns
        np.testing.assert_array_equal(out["z"].to_numpy(), np.asarray([7.0, 8.0]))

    def test_row_count_matches_input(self) -> None:
        selected = _df(COORDS)
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit)
        assert out.shape[0] == 2

    def test_custom_name(self) -> None:
        selected = _df(COORDS)
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
            name="dist_to_station",
        )
        out = apply_geo_distance(selected, fit)
        assert out.columns == ["dist_to_station"]

    def test_keep_original_cols_retains_inputs(self) -> None:
        selected = _df(COORDS)
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit, keep_original_cols=True)
        assert set(out.columns) == {"lat", "lon", "geo_dist"}


# ---------------------------------------------------------------------------
# apply_geo_distance — haversine distance values
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_row_at_reference_point_has_zero_distance(self) -> None:
        selected = _df({"lat": [UNION_STATION[0]], "lon": [UNION_STATION[1]]})
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit)
        assert out["geo_dist"][0] == pytest.approx(0.0, abs=1e-6)

    def test_distance_is_positive_meters_for_known_pair(self) -> None:
        # Union Station -> Smithsonian Castle is roughly 1.9km.
        selected = _df({"lat": [38.888504], "lon": [-77.026352]})
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit)
        assert 1500.0 < out["geo_dist"][0] < 2500.0

    def test_symmetry(self) -> None:
        # distance(A -> B) == distance(B -> A).
        a = (38.8986312, -77.0062457)
        b = (38.888504, -77.026352)
        forward = apply_geo_distance(
            _df({"lat": [b[0]], "lon": [b[1]]}),
            fit_geo_distance(
                _df({"lat": [b[0]], "lon": [b[1]]}),
                lat="lat",
                lon="lon",
                ref_lat=a[0],
                ref_lon=a[1],
            ),
        )
        backward = apply_geo_distance(
            _df({"lat": [a[0]], "lon": [a[1]]}),
            fit_geo_distance(
                _df({"lat": [a[0]], "lon": [a[1]]}),
                lat="lat",
                lon="lon",
                ref_lat=b[0],
                ref_lon=b[1],
            ),
        )
        assert forward["geo_dist"][0] == pytest.approx(backward["geo_dist"][0])


# ---------------------------------------------------------------------------
# apply_geo_distance — euclidean distance values
# ---------------------------------------------------------------------------


class TestEuclidean:
    def test_distance_matches_hand_calc(self) -> None:
        selected = _df({"lat": [3.0], "lon": [4.0]})
        fit = fit_geo_distance(
            selected, lat="lat", lon="lon", ref_lat=0.0, ref_lon=0.0, metric="euclidean"
        )
        out = apply_geo_distance(selected, fit)
        # sqrt(3^2 + 4^2) = 5.
        assert out["geo_dist"][0] == pytest.approx(5.0)

    def test_row_at_reference_point_has_zero_distance(self) -> None:
        selected = _df({"lat": [1.0], "lon": [2.0]})
        fit = fit_geo_distance(
            selected, lat="lat", lon="lon", ref_lat=1.0, ref_lon=2.0, metric="euclidean"
        )
        out = apply_geo_distance(selected, fit)
        assert out["geo_dist"][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_geo_distance — log_distance
# ---------------------------------------------------------------------------


class TestLogDistance:
    def test_log_value_matches_log_of_raw(self) -> None:
        selected = _df({"lat": [38.888504], "lon": [-77.026352]})
        raw = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        logged = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
            log_distance=True,
        )
        raw_out = apply_geo_distance(selected, raw)
        log_out = apply_geo_distance(selected, logged)
        assert log_out["geo_dist"][0] == pytest.approx(math.log(raw_out["geo_dist"][0]))


# ---------------------------------------------------------------------------
# apply_geo_distance — missing-value propagation
# ---------------------------------------------------------------------------


class TestApplyMissingValues:
    def test_nan_lat_produces_nan_distance(self) -> None:
        selected = _df({"lat": [float("nan"), 38.888504], "lon": [-77.006, -77.026352]})
        fit = fit_geo_distance(
            selected,
            lat="lat",
            lon="lon",
            ref_lat=UNION_STATION[0],
            ref_lon=UNION_STATION[1],
        )
        out = apply_geo_distance(selected, fit)
        assert math.isnan(float(out["geo_dist"][0]))
        assert math.isfinite(float(out["geo_dist"][1]))
