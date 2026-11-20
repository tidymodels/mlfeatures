"""Geographic distance feature: distance from each row to a fixed reference point.

The *fit* half (`fit_geo_distance()`) does no estimation from data (the
reference point and distance settings are all constructor arguments, not
learned statistics), so it just validates and packages them alongside the
resolved `lat`/`lon` column names. The *apply* half
(`apply_geo_distance()`) computes the per-row distance. `StepGeoDistance`
wraps them into the `Step` lifecycle so a `Recipe` can fit and apply it.

See `mlfeatures.step_interact` for the precedent of a fit function with nothing
to estimate, and `mlfeatures.step_class_distance` for the precedent of a
distance-to-a-point step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt

from mlfeatures import _frame
from mlfeatures.step import Step
from mlfeatures.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = [
    "GeoDistanceFit",
    "GeoMetric",
    "StepGeoDistance",
    "apply_geo_distance",
    "fit_geo_distance",
]


# ``Literal`` makes the allowed metric values part of the type so a typo at the
# call site is caught by mypy/pyright. Add a new metric here when ready to
# support it: the runtime dispatch in ``_compute_distance`` and the relevant
# tests are what need to change in lockstep.
GeoMetric = Literal["haversine", "euclidean"]

_EARTH_RADIUS_METERS = 6371e3


@dataclass(frozen=True)
class GeoDistanceFit:
    """The fitted result returned by `fit_geo_distance()`.

    Attributes
    ----------
    lat
        Name of the latitude column.
    lon
        Name of the longitude column.
    ref_lat
        Latitude of the reference point.
    ref_lon
        Longitude of the reference point.
    metric
        Distance function to apply: ``"haversine"`` or ``"euclidean"``.
    log_distance
        Whether `apply_geo_distance()` should return ``log(distance)``
        rather than the raw distance.
    name
        Name of the output distance column.
    """

    lat: str
    lon: str
    ref_lat: float
    ref_lon: float
    metric: GeoMetric
    log_distance: bool
    name: str


def fit_geo_distance(
    selected: Frame,
    *,
    lat: str,
    lon: str,
    ref_lat: float,
    ref_lon: float,
    metric: GeoMetric = "haversine",
    log_distance: bool = False,
    name: str = "geo_dist",
) -> GeoDistanceFit:
    """Package the reference point and distance settings for `apply_geo_distance()`.

    There is nothing to estimate from ``selected``: the reference point is a
    fixed constructor argument, not a learned statistic. This function exists
    so the fit/apply split matches every other step, and so the data frame is
    available if a future version wants to validate observed lat/lon ranges
    against the training data.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the ``lat`` and ``lon`` columns.
    lat
        Name of the latitude column.
    lon
        Name of the longitude column.
    ref_lat
        Latitude of the reference point.
    ref_lon
        Longitude of the reference point.
    metric
        Distance function to use at apply time: ``"haversine"`` (default) for
        great-circle distance in meters, or ``"euclidean"`` for the flat-plane
        distance between the raw ``lat``/``lon`` values.
    log_distance
        If ``True``, `apply_geo_distance()` returns ``log(distance)``.
        Default ``False``.
    name
        Name of the output distance column. Default ``"geo_dist"``.

    Returns
    -------
    GeoDistanceFit
        Fitted object containing the reference point and distance
        configuration. See `GeoDistanceFit` for the field-by-field layout.
    """
    # ``selected`` is unused beyond confirming the columns exist at fit time;
    # kept as a parameter to match the fit/apply pattern used across steps.
    del selected
    return GeoDistanceFit(
        lat=lat,
        lon=lon,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
        metric=metric,
        log_distance=log_distance,
        name=name,
    )


def apply_geo_distance(
    data: Frame,
    fit_result: GeoDistanceFit,
    *,
    keep_original_cols: bool = False,
) -> Frame:
    """Add the per-row distance from ``fit_result``'s reference point.

    Computes the distance from each row's ``(lat, lon)`` to
    ``(fit_result.ref_lat, fit_result.ref_lon)`` and adds it as a column
    named ``fit_result.name``. The ``lat``/``lon`` columns are dropped unless
    ``keep_original_cols``; every other column of ``data`` passes through
    unchanged.

    Parameters
    ----------
    data
        Internal `Frame` to transform.
    fit_result
        Output of `fit_geo_distance()`.
    keep_original_cols
        When ``True``, the ``lat``/``lon`` columns are kept alongside the
        distance column instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with one new ``Float64`` distance column. The row count is
        preserved.

    Notes
    -----
    NaN propagation: a null in either ``lat`` or ``lon`` produces a null
    distance for that row.
    """
    lat_arr = _frame.series_to_float_array(data[fit_result.lat])
    lon_arr = _frame.series_to_float_array(data[fit_result.lon])

    distance = _compute_distance(
        lat_arr, lon_arr, fit_result.ref_lat, fit_result.ref_lon, fit_result.metric
    )
    if fit_result.log_distance:
        distance = np.log(distance)

    data = data.with_columns(
        nw.new_series(
            name=fit_result.name, values=distance, backend=data.implementation
        )
    )
    if not keep_original_cols:
        data = data.drop([fit_result.lat, fit_result.lon])
    return data


# --- Private helpers --------------------------------------------------------


def _compute_distance(
    lat: npt.NDArray[np.float64],
    lon: npt.NDArray[np.float64],
    ref_lat: float,
    ref_lon: float,
    metric: GeoMetric,
) -> npt.NDArray[np.float64]:
    """Distance from each ``(lat, lon)`` row to the reference point.

    Parameters
    ----------
    lat, lon
        Coordinate arrays of shape ``(n,)``.
    ref_lat, ref_lon
        Reference point coordinates.
    metric
        Which distance function to compute.

    Returns
    -------
    ndarray
        Float-64 array of shape ``(n,)``.

    Notes
    -----
    ``"haversine"`` uses the great-circle formula (result in meters);
    ``"euclidean"`` is the flat-plane distance between the raw ``lat``/``lon``
    values, matching R recipes' ``is_lat_lon = FALSE`` mode: it does not
    correct for longitude degrees covering less real-world distance at higher
    latitudes, so it is only meaningful as a rough, unit-free distance.
    """
    if metric == "euclidean":
        result = np.sqrt((lon - ref_lon) ** 2 + (lat - ref_lat) ** 2)
        return np.asarray(result, dtype=np.float64)
    if metric == "haversine":
        to_rad = math.pi / 180.0
        lat_rad = lat * to_rad
        ref_lat_rad = ref_lat * to_rad
        delta_lat = (ref_lat_rad - lat_rad) / 2.0
        delta_lon = (ref_lon * to_rad - lon * to_rad) / 2.0
        a = (
            np.sin(delta_lat) ** 2
            + np.cos(lat_rad) * np.cos(ref_lat_rad) * np.sin(delta_lon) ** 2
        )
        c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        result = _EARTH_RADIUS_METERS * c
        return np.asarray(result, dtype=np.float64)
    # The Literal type means mypy/pyright already catch typos, but a runtime
    # guard helps callers passing string literals from untyped sources (CLI,
    # YAML configs, etc.).
    raise ValueError(  # pragma: no cover
        f"unknown metric {metric!r}; expected 'haversine' or 'euclidean'"
    )


class StepGeoDistance(Step):
    """Add the distance from each row to a fixed reference point.

    Replaces the ``lat``/``lon`` columns with a single ``name`` column
    holding each row's distance to ``(ref_lat, ref_lon)``. With the default
    ``"haversine"`` metric the result is the great-circle distance in meters;
    with ``"euclidean"`` it is the flat-plane distance between the raw
    ``lat``/``lon`` values.

    See [Spatial Distance](https://feaz-book.com/spatial-distance) in *Feature
    Engineering A to Z* for background on this technique.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    lat
        Name of the latitude column.
    lon
        Name of the longitude column.
    ref_lat
        Latitude of the reference point. Must be between -90 and 90 when
        ``metric`` is ``"haversine"``.
    ref_lon
        Longitude of the reference point. Must be between -180 and 180 when
        ``metric`` is ``"haversine"``.
    metric
        Distance function: ``"haversine"`` (default) for great-circle
        distance in meters, or ``"euclidean"`` for the flat-plane distance
        between the raw ``lat``/``lon`` values.
    log_distance
        Whether to return ``log(distance)`` instead of the raw distance. A
        row exactly at the reference point has distance 0, which gives
        ``-inf``; there is no ``eps`` offset as in ``StepClassDistance``.
        Default ``False``.
    name
        Name of the output distance column. Default ``"geo_dist"``. A
        `Recipe` rejects the step if the data frame already has a column of this
        name.
    keep_original_cols
        If ``True``, the ``lat``/``lon`` columns are kept alongside the
        distance column instead of being dropped. Default ``False``.
    role
        Role assigned to the new distance column (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"geo_distance"``; a second step
        of this type in the same recipe becomes ``"geo_distance_2"``, and
        passing an ``id`` already used in the recipe is an error.

    Examples
    --------
    Each row's distance to Union Station in Washington, D.C.:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepGeoDistance
    ```

    ```{python}
    df = pd.DataFrame(
        {
            "name": ["Union Station", "Smithsonian Castle"],
            "lat": [38.897303, 38.888504],
            "lon": [-77.006430, -77.026352],
        }
    )
    df
    ```

    Specifying the step touches no data, so the recipe reports itself as
    untrained:

    ```{python}
    recipe = Recipe().add(
        StepGeoDistance("lat", "lon", ref_lat=38.8986312, ref_lon=-77.0062457)
    )
    recipe
    ```

    `fit()` records the reference point and returns a trained recipe:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces ``lat``/``lon`` with the ``geo_dist`` column, in
    meters:

    ```{python}
    fitted.transform(df)
    ```

    ``metric="euclidean"`` computes the flat-plane distance between the raw
    coordinate values instead:

    ```{python}
    recipe = Recipe().add(
        StepGeoDistance(
            "lat", "lon", ref_lat=38.8986312, ref_lon=-77.0062457, metric="euclidean"
        )
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps ``lat``/``lon`` alongside the distance:

    ```{python}
    recipe = Recipe().add(
        StepGeoDistance(
            "lat",
            "lon",
            ref_lat=38.8986312,
            ref_lon=-77.0062457,
            keep_original_cols=True,
        )
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.MANY_TO_MANY

    #: Populated by ``fit()``: the latitude and longitude columns, the reference
    #: point, and the distance settings.
    _estimates: GeoDistanceFit | None

    def __init__(
        self,
        lat: str,
        lon: str,
        *,
        ref_lat: float,
        ref_lon: float,
        metric: GeoMetric = "haversine",
        log_distance: bool = False,
        name: str = "geo_dist",
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        if metric == "haversine":
            if not -90.0 <= ref_lat <= 90.0:
                raise ValueError(
                    f"'ref_lat' must be between -90 and 90; got {ref_lat!r}"
                )
            if not -180.0 <= ref_lon <= 180.0:
                raise ValueError(
                    f"'ref_lon' must be between -180 and 180; got {ref_lon!r}"
                )
        self.lat = lat
        self.lon = lon
        self.columns = [lat, lon]
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        # Pin the attribute type to ``GeoMetric`` so type-checkers don't widen
        # the stored value back to ``str`` and complain when we hand it to
        # ``fit_geo_distance(metric=...)`` later.
        self.metric: GeoMetric = metric
        self.log_distance = log_distance
        self.name = name
        super().__init__(
            id,
            "geo_distance",
            keep_original_cols=keep_original_cols,
            output_role=role,
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Package the reference point and validate the lat/lon columns exist."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        self._estimates = fit_geo_distance(
            selected,
            lat=self.lat,
            lon=self.lon,
            ref_lat=self.ref_lat,
            ref_lon=self.ref_lon,
            metric=self.metric,
            log_distance=self.log_distance,
            name=self.name,
        )
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        self._schema_out = ColSchema(columns=[self.name], dtypes={self.name: "Float64"})
        self._output_parents = {self.name: [self.lat, self.lon]}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Add the distance column. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepGeoDistance.transform() called before fit()")
        return apply_geo_distance(
            new_data, self._estimates, keep_original_cols=self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the ``lat``/``lon`` columns consumed at fit time."""
        if self._schema_in is None:
            raise RuntimeError("StepGeoDistance.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the new distance column produced by ``transform``."""
        if self._schema_out is None:
            raise RuntimeError("StepGeoDistance.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """The distance column is derived from both ``lat`` and ``lon``."""
        if self._output_parents is None:
            raise RuntimeError("StepGeoDistance.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Distances are only meaningful on numeric lat/lon columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        if self._output_parents is None:
            return
        if self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)
        if self.keep_original_cols:
            return
        for col in (self.lat, self.lon):
            for role in list(roles.of(col)):
                roles.remove(col, role)

    @property
    def produces_sparse(self) -> bool:
        """Distances are dense by construction: no zeros structurally."""
        return False
