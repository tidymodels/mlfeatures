"""Conformance tests for StepGeoDistance.

The step contract itself lives in `shoez.testing.StepConformanceTest`; this file
only supplies the step and the data to run it against. Behaviour specific to
StepGeoDistance belongs in ``test_step_geo_distance.py`` and ``test_step_geo_distance_e2e.py``.
"""

from __future__ import annotations

from shoez import StepGeoDistance
from shoez.testing import StepConformanceTest


class TestGeoDistanceConformance(StepConformanceTest):
    training_data = {
        "lat": [38.897303, 38.888504, 38.891661],
        "lon": [-77.006430, -77.026352, -77.049567],
    }

    def make_step(self) -> StepGeoDistance:
        return StepGeoDistance("lat", "lon", ref_lat=38.8986312, ref_lon=-77.0062457)
