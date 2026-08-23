"""Tests for fit_class_distance and apply_class_distance.

Mirrors the structure of ``tests/test_step_normalize.py`` and
``tests/test_step_polynomial.py``: pytest with class-based grouping, a tiny
``_df`` helper, narwhals frames built from pandas. The functions under test
operate on the internal narwhals :data:`~shoez.types.Frame` type; the public
contract is narwhals-in, narwhals-out and is backend-agnostic by construction,
so we don't parametrise pandas vs polars at this layer (the end-to-end test
does that for the ``StepClassDistance`` wrapper).
"""

from __future__ import annotations

import math
from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from shoez.step_class_distance import (
    ClassDistanceFit,
    apply_class_distance,
    fit_class_distance,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    """Build a narwhals frame from a plain dict (pandas backend)."""
    return nw.from_native(pd.DataFrame(data), eager_only=True)


def _series(values: list[Any]) -> nw.Series[Any]:
    """Build a narwhals Series from a plain list (pandas backend)."""
    return nw.from_native(pd.Series(values), series_only=True)


# A small two-class, two-feature example reused across several tests. The
# centroids by hand:
#   class 0 (rows 0, 1): x=[0, 2], y=[0, 4] -> centroid (1, 2)
#   class 1 (rows 2, 3): x=[10, 12], y=[10, 14] -> centroid (11, 12)
SMALL_FEATURES: dict[str, list[float]] = {
    "x": [0.0, 2.0, 10.0, 12.0],
    "y": [0.0, 4.0, 10.0, 14.0],
}
SMALL_TARGET: list[int] = [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# fit_class_distance — return type & shape
# ---------------------------------------------------------------------------


class TestFitReturnType:
    """`fit_class_distance` returns a `ClassDistanceFit` with the documented shape."""

    def test_returns_class_distance_fit(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        assert isinstance(result, ClassDistanceFit)

    def test_centroids_dtype_is_float64(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        assert result.centroids.dtype == np.float64

    def test_centroids_shape_is_p_by_c(self) -> None:
        # 2 features, 2 classes -> (2, 2).
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        assert result.centroids.shape == (2, 2)

    def test_lengths_match_columns_and_classes(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        assert len(result.columns) == result.centroids.shape[0]
        assert len(result.classes) == result.centroids.shape[1]


# ---------------------------------------------------------------------------
# fit_class_distance — centroid values
# ---------------------------------------------------------------------------


class TestFitCentroids:
    """Per-class column-wise means line up with hand calculations."""

    def test_two_classes_two_features(self) -> None:
        # Hand calculation in module header.
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        # Row 0 (feature "x"): class 0 = 1.0, class 1 = 11.0.
        np.testing.assert_allclose(result.centroids[0], [1.0, 11.0])
        # Row 1 (feature "y"): class 0 = 2.0, class 1 = 12.0.
        np.testing.assert_allclose(result.centroids[1], [2.0, 12.0])

    def test_string_target_labels(self) -> None:
        result = fit_class_distance(
            _df(SMALL_FEATURES),
            _series(["red", "red", "blue", "blue"]),
        )
        # "blue" sorts before "red", so the column order is (blue, red).
        assert result.classes == ["blue", "red"]
        # "blue" rows are 2, 3 -> centroid (11, 12).
        np.testing.assert_allclose(result.centroids[:, 0], [11.0, 12.0])
        # "red" rows are 0, 1 -> centroid (1, 2).
        np.testing.assert_allclose(result.centroids[:, 1], [1.0, 2.0])

    def test_integer_target_labels(self) -> None:
        # Integer labels sort numerically.
        result = fit_class_distance(_df(SMALL_FEATURES), _series([2, 2, 5, 5]))
        assert result.classes == [2, 5]
        np.testing.assert_allclose(result.centroids[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(result.centroids[:, 1], [11.0, 12.0])

    def test_three_classes(self) -> None:
        # Three single-row "classes" — each centroid is just that row.
        selected = _df({"x": [10.0, 20.0, 30.0], "y": [100.0, 200.0, 300.0]})
        result = fit_class_distance(selected, _series(["a", "b", "c"]))
        assert result.classes == ["a", "b", "c"]
        # Feature x across classes a, b, c: 10, 20, 30.
        np.testing.assert_allclose(result.centroids[0], [10.0, 20.0, 30.0])
        # Feature y across classes a, b, c: 100, 200, 300.
        np.testing.assert_allclose(result.centroids[1], [100.0, 200.0, 300.0])


# ---------------------------------------------------------------------------
# fit_class_distance — ordering & determinism
# ---------------------------------------------------------------------------


class TestFitOrdering:
    """`columns` follows input column order; `classes` is sorted."""

    def test_class_order_is_sorted(self) -> None:
        # Inject classes in reverse alphabetical order in the target.
        result = fit_class_distance(_df(SMALL_FEATURES), _series(["z", "z", "a", "a"]))
        # Sorted ascending: ["a", "z"].
        assert result.classes == ["a", "z"]

    def test_column_order_follows_input(self) -> None:
        # Build the features in a deliberately scrambled order.
        selected = _df({"c": [0.0, 1.0], "a": [10.0, 11.0], "b": [100.0, 101.0]})
        result = fit_class_distance(selected, _series([0, 1]))
        assert result.columns == ["c", "a", "b"]

    def test_row_indices_map_to_columns(self) -> None:
        # Two classes, three features -> centroids[i] is feature i.
        selected = _df({"p0": [1.0, 9.0], "p1": [2.0, 8.0], "p2": [3.0, 7.0]})
        result = fit_class_distance(selected, _series([0, 1]))
        # Class 0 row is [1, 2, 3] -> centroids[:, 0] == [1, 2, 3].
        np.testing.assert_allclose(result.centroids[:, 0], [1.0, 2.0, 3.0])
        # Class 1 row is [9, 8, 7] -> centroids[:, 1] == [9, 8, 7].
        np.testing.assert_allclose(result.centroids[:, 1], [9.0, 8.0, 7.0])


# ---------------------------------------------------------------------------
# fit_class_distance — null target handling
# ---------------------------------------------------------------------------


class TestFitNullTarget:
    """Rows whose target is null are dropped before centroid means are taken."""

    def test_nan_target_row_excluded(self) -> None:
        # A null label among class-0 rows would normally pull the centroid
        # toward whatever feature values shared its row. Dropping it first
        # leaves only the genuine class-0 row, so the centroid equals that row.
        selected = _df({"x": [0.0, 999.0, 10.0, 12.0]})
        # Index 1 has an unlabelled target (NaN). Index 0 is the only true
        # class-0 row; index 2/3 are class 1.
        target = _series([0.0, float("nan"), 1.0, 1.0])
        result = fit_class_distance(selected, target)
        # Class 0 centroid is just the row at index 0: x = 0.
        # (If the NaN row had been included, the mean would have been ~500.)
        np.testing.assert_allclose(result.centroids[:, 0], [0.0])

    def test_classes_unaffected_by_null_targets(self) -> None:
        # The NaN row shouldn't introduce a "NaN class".
        selected = _df({"x": [0.0, 999.0, 10.0]})
        target = _series([0.0, float("nan"), 1.0])
        result = fit_class_distance(selected, target)
        # Only two real classes, sorted: [0.0, 1.0].
        assert result.classes == [0.0, 1.0]


# ---------------------------------------------------------------------------
# apply_class_distance — output shape & passthrough
# ---------------------------------------------------------------------------


class TestApplyShape:
    """The fitted input columns are replaced by exactly `c` new columns."""

    def test_replaces_inputs_with_c_new_columns(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET))
        out = apply_class_distance(selected, fit)
        # The 2 input columns are gone; 2 class-distance columns take their place.
        assert out.shape == (4, 2)

    def test_input_columns_dropped(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET))
        out = apply_class_distance(selected, fit)
        assert set(out.columns) == {"distance_0", "distance_1"}

    def test_unselected_columns_pass_through_unchanged(self) -> None:
        data = _df({**SMALL_FEATURES, "z": [7.0, 8.0, 9.0, 10.0]})
        selected = data.select(["x", "y"])
        fit = fit_class_distance(selected, _series(SMALL_TARGET))
        out = apply_class_distance(data, fit)
        assert "z" in out.columns
        np.testing.assert_array_equal(
            out["z"].to_numpy(), np.asarray([7.0, 8.0, 9.0, 10.0])
        )

    def test_row_count_matches_input(self) -> None:
        selected = _df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        target = _series(["a", "a", "b", "b", "a"])
        fit = fit_class_distance(selected, target)
        out = apply_class_distance(selected, fit)
        assert out.shape[0] == 5

    def test_integer_features_yield_float_distance_columns(self) -> None:
        # class_distance accepts integer feature columns, but distances are
        # real-valued — the new columns must be float even when the inputs are
        # int and a distance could coincidentally land on a whole number.
        selected = _df({"x": [0, 2, 10, 12], "y": [0, 4, 10, 14]})
        assert selected["x"].dtype.is_integer()  # integer in
        fit = fit_class_distance(selected, _series(SMALL_TARGET))
        out = apply_class_distance(selected, fit)
        for col in ("distance_0", "distance_1"):
            assert out[col].dtype in (nw.Float32, nw.Float64)


# ---------------------------------------------------------------------------
# apply_class_distance — column naming
# ---------------------------------------------------------------------------


class TestApplyColumnNames:
    """New columns follow the `{prefix}_{class}` naming convention."""

    def test_default_prefix(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(["alpha", "alpha", "beta", "beta"]))
        out = apply_class_distance(selected, fit)
        assert out.columns[-2:] == ["distance_alpha", "distance_beta"]

    def test_custom_prefix(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(["a", "a", "b", "b"]))
        out = apply_class_distance(selected, fit, prefix="dist")
        assert out.columns[-2:] == ["dist_a", "dist_b"]

    def test_integer_class_labels_format_cleanly(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series([3, 3, 7, 7]))
        out = apply_class_distance(selected, fit)
        assert out.columns[-2:] == ["distance_3", "distance_7"]


# ---------------------------------------------------------------------------
# apply_class_distance — Euclidean distance values
# ---------------------------------------------------------------------------


class TestApplyDistancesEuclidean:
    """With log_distance=False, raw Euclidean distances match hand calc."""

    def test_row_on_centroid_has_zero_distance(self) -> None:
        # Single row per class -> each row equals its own centroid -> zero.
        selected = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(selected, _series(["a", "b"]), log_distance=False)
        out = apply_class_distance(selected, fit)
        # Row 0 (class "a") is at its own centroid for class "a".
        assert out["distance_a"][0] == pytest.approx(0.0)
        # Row 1 (class "b") is at its own centroid for class "b".
        assert out["distance_b"][1] == pytest.approx(0.0)

    def test_distance_matches_hand_calc(self) -> None:
        # Centroids by hand: class 0 -> (1, 2); class 1 -> (11, 12).
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        out = apply_class_distance(selected, fit)
        # Row 0 is (0, 0): distance to (1, 2) = sqrt(1 + 4) = sqrt(5).
        assert out["distance_0"][0] == pytest.approx(math.sqrt(5.0))
        # Row 0 to centroid (11, 12) = sqrt(121 + 144) = sqrt(265).
        assert out["distance_1"][0] == pytest.approx(math.sqrt(265.0))


# ---------------------------------------------------------------------------
# apply_class_distance — Manhattan metric
# ---------------------------------------------------------------------------


class TestMetricManhattan:
    """L1 distance: sum of absolute differences along the feature axis."""

    def test_row_on_centroid_has_zero_distance(self) -> None:
        selected = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(
            selected,
            _series(["a", "b"]),
            metric="manhattan",
            log_distance=False,
        )
        out = apply_class_distance(selected, fit)
        assert out["distance_a"][0] == pytest.approx(0.0)
        assert out["distance_b"][1] == pytest.approx(0.0)

    def test_distance_matches_hand_calc(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(
            selected,
            _series(SMALL_TARGET),
            metric="manhattan",
            log_distance=False,
        )
        out = apply_class_distance(selected, fit)
        # Row 0 is (0, 0): |0 - 1| + |0 - 2| = 3 (to class 0); |0 - 11| + |0 - 12| = 23 (to class 1).
        assert out["distance_0"][0] == pytest.approx(3.0)
        assert out["distance_1"][0] == pytest.approx(23.0)

    def test_metric_stored_on_fit_object(self) -> None:
        # Sanity: the metric is what we asked for, not silently flipped.
        fit = fit_class_distance(
            _df(SMALL_FEATURES),
            _series(SMALL_TARGET),
            metric="manhattan",
        )
        assert fit.metric == "manhattan"


# ---------------------------------------------------------------------------
# apply_class_distance — log_distance and eps
# ---------------------------------------------------------------------------


class TestLogDistance:
    """`log_distance` returns log(raw + eps); `eps` keeps the log finite."""

    def test_log_default_is_true(self) -> None:
        fit = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_TARGET))
        assert fit.log_distance is True

    def test_log_value_matches_log_of_raw_plus_eps(self) -> None:
        selected = _df(SMALL_FEATURES)
        raw = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        logged = fit_class_distance(selected, _series(SMALL_TARGET))  # default log=True
        raw_out = apply_class_distance(selected, raw)
        log_out = apply_class_distance(selected, logged)
        # For every row & class, log_out == log(raw_out + eps).
        for cls in (0, 1):
            col = f"distance_{cls}"
            np.testing.assert_allclose(
                log_out[col].to_numpy(),
                np.log(raw_out[col].to_numpy() + 0.001),
            )

    def test_row_on_centroid_is_finite_with_default_eps(self) -> None:
        # Distance 0 + eps 0.001 -> log(0.001), finite and < 0.
        selected = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(selected, _series(["a", "b"]))
        out = apply_class_distance(selected, fit)
        zero_dist_log = float(out["distance_a"][0])
        assert math.isfinite(zero_dist_log)
        assert zero_dist_log == pytest.approx(math.log(0.001))

    def test_custom_eps_honored(self) -> None:
        selected = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(selected, _series(["a", "b"]), eps=0.5)
        out = apply_class_distance(selected, fit)
        # Row 0 has raw distance 0 to its own centroid; log(0 + 0.5) = log(0.5).
        assert float(out["distance_a"][0]) == pytest.approx(math.log(0.5))


# ---------------------------------------------------------------------------
# apply_class_distance — missing-value propagation
# ---------------------------------------------------------------------------


class TestApplyMissingValues:
    """A NaN feature in a row makes that row's distances NaN, others unaffected."""

    def test_nan_feature_row_becomes_nan(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        # Build a new frame with a NaN in row 1.
        with_nan = _df(
            {"x": [0.0, float("nan"), 10.0, 12.0], "y": [0.0, 4.0, 10.0, 14.0]}
        )
        out = apply_class_distance(with_nan, fit)
        # Row 1's distances should be NaN.
        assert math.isnan(float(out["distance_0"][1]))
        assert math.isnan(float(out["distance_1"][1]))

    def test_other_rows_unaffected_by_nan(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        with_nan = _df(
            {"x": [0.0, float("nan"), 10.0, 12.0], "y": [0.0, 4.0, 10.0, 14.0]}
        )
        out = apply_class_distance(with_nan, fit)
        # Rows 0, 2, 3 must be finite — the NaN in row 1 doesn't propagate.
        for row in (0, 2, 3):
            assert math.isfinite(float(out["distance_0"][row]))
            assert math.isfinite(float(out["distance_1"][row]))


# ---------------------------------------------------------------------------
# Round trip & non-feature passthrough
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end fit + apply on a small dataset; basic sanity invariants."""

    def test_distances_non_negative_and_finite(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        out = apply_class_distance(selected, fit)
        for col in ("distance_0", "distance_1"):
            arr = out[col].to_numpy()
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)

    def test_single_row_class_has_zero_self_distance(self) -> None:
        # When a class has exactly one row, the centroid is that row, so the
        # row's distance to its own class centroid is exactly 0.
        selected = _df({"x": [1.0, 5.0, 9.0]})
        fit = fit_class_distance(selected, _series(["a", "b", "c"]), log_distance=False)
        out = apply_class_distance(selected, fit)
        assert float(out["distance_a"][0]) == pytest.approx(0.0)
        assert float(out["distance_b"][1]) == pytest.approx(0.0)
        assert float(out["distance_c"][2]) == pytest.approx(0.0)

    def test_keep_original_cols_retains_inputs(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        out = apply_class_distance(selected, fit, keep_original_cols=True)
        assert set(out.columns) == set(SMALL_FEATURES) | {"distance_0", "distance_1"}
        for name, values in SMALL_FEATURES.items():
            np.testing.assert_array_equal(out[name].to_numpy(), np.asarray(values))

    def test_keep_original_cols_does_not_change_the_distances(self) -> None:
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        dropped = apply_class_distance(selected, fit)
        kept = apply_class_distance(selected, fit, keep_original_cols=True)
        np.testing.assert_allclose(
            kept["distance_0"].to_numpy(), dropped["distance_0"].to_numpy()
        )

    def test_extra_passthrough_column_kept(self) -> None:
        # An unrelated column on ``data`` should survive apply unchanged.
        selected = _df(SMALL_FEATURES)
        fit = fit_class_distance(selected, _series(SMALL_TARGET), log_distance=False)
        with_extra = _df({**SMALL_FEATURES, "passthrough": [99.0, 99.0, 99.0, 99.0]})
        out = apply_class_distance(with_extra, fit)
        # Extra column survives and keeps its values.
        assert "passthrough" in out.columns
        np.testing.assert_array_equal(
            out["passthrough"].to_numpy(), np.asarray([99.0, 99.0, 99.0, 99.0])
        )
