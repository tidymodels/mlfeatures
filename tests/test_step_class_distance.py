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


# A small two-class, two-predictor example reused across several tests. The
# centroids by hand:
#   class 0 (rows 0, 1): x=[0, 2], y=[0, 4] -> centroid (1, 2)
#   class 1 (rows 2, 3): x=[10, 12], y=[10, 14] -> centroid (11, 12)
SMALL_FEATURES: dict[str, list[float]] = {
    "x": [0.0, 2.0, 10.0, 12.0],
    "y": [0.0, 4.0, 10.0, 14.0],
}
SMALL_OUTCOME: list[int] = [0, 0, 1, 1]


# ---------------------------------------------------------------------------
# fit_class_distance — return type & shape
# ---------------------------------------------------------------------------


class TestFitReturnType:
    """`fit_class_distance` returns a `ClassDistanceFit` with the documented shape."""

    def test_returns_class_distance_fit(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        assert isinstance(result, ClassDistanceFit)

    def test_centroids_dtype_is_float64(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        assert result.centroids.dtype == np.float64

    def test_centroids_shape_is_p_by_c(self) -> None:
        # 2 predictors, 2 classes -> (2, 2).
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        assert result.centroids.shape == (2, 2)

    def test_lengths_match_columns_and_classes(self) -> None:
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        assert len(result.columns) == result.centroids.shape[0]
        assert len(result.classes) == result.centroids.shape[1]


# ---------------------------------------------------------------------------
# fit_class_distance — centroid values
# ---------------------------------------------------------------------------


class TestFitCentroids:
    """Per-class column-wise means line up with hand calculations."""

    def test_two_classes_two_predictors(self) -> None:
        # Hand calculation in module header.
        result = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        # Row 0 (predictor "x"): class 0 = 1.0, class 1 = 11.0.
        np.testing.assert_allclose(result.centroids[0], [1.0, 11.0])
        # Row 1 (predictor "y"): class 0 = 2.0, class 1 = 12.0.
        np.testing.assert_allclose(result.centroids[1], [2.0, 12.0])

    def test_string_outcome_labels(self) -> None:
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

    def test_integer_outcome_labels(self) -> None:
        # Integer labels sort numerically.
        result = fit_class_distance(_df(SMALL_FEATURES), _series([2, 2, 5, 5]))
        assert result.classes == [2, 5]
        np.testing.assert_allclose(result.centroids[:, 0], [1.0, 2.0])
        np.testing.assert_allclose(result.centroids[:, 1], [11.0, 12.0])

    def test_three_classes(self) -> None:
        # Three single-row "classes" — each centroid is just that row.
        features = _df({"x": [10.0, 20.0, 30.0], "y": [100.0, 200.0, 300.0]})
        result = fit_class_distance(features, _series(["a", "b", "c"]))
        assert result.classes == ["a", "b", "c"]
        # Predictor x across classes a, b, c: 10, 20, 30.
        np.testing.assert_allclose(result.centroids[0], [10.0, 20.0, 30.0])
        # Predictor y across classes a, b, c: 100, 200, 300.
        np.testing.assert_allclose(result.centroids[1], [100.0, 200.0, 300.0])


# ---------------------------------------------------------------------------
# fit_class_distance — ordering & determinism
# ---------------------------------------------------------------------------


class TestFitOrdering:
    """`columns` follows input column order; `classes` is sorted."""

    def test_class_order_is_sorted(self) -> None:
        # Inject classes in reverse alphabetical order in the outcome.
        result = fit_class_distance(_df(SMALL_FEATURES), _series(["z", "z", "a", "a"]))
        # Sorted ascending: ["a", "z"].
        assert result.classes == ["a", "z"]

    def test_column_order_follows_input(self) -> None:
        # Build the predictors in a deliberately scrambled order.
        features = _df({"c": [0.0, 1.0], "a": [10.0, 11.0], "b": [100.0, 101.0]})
        result = fit_class_distance(features, _series([0, 1]))
        assert result.columns == ["c", "a", "b"]

    def test_row_indices_map_to_columns(self) -> None:
        # Two classes, three predictors -> centroids[i] is predictor i.
        features = _df({"p0": [1.0, 9.0], "p1": [2.0, 8.0], "p2": [3.0, 7.0]})
        result = fit_class_distance(features, _series([0, 1]))
        # Class 0 row is [1, 2, 3] -> centroids[:, 0] == [1, 2, 3].
        np.testing.assert_allclose(result.centroids[:, 0], [1.0, 2.0, 3.0])
        # Class 1 row is [9, 8, 7] -> centroids[:, 1] == [9, 8, 7].
        np.testing.assert_allclose(result.centroids[:, 1], [9.0, 8.0, 7.0])


# ---------------------------------------------------------------------------
# fit_class_distance — null outcome handling
# ---------------------------------------------------------------------------


class TestFitNullOutcome:
    """Rows whose outcome is null are dropped before centroid means are taken."""

    def test_nan_outcome_row_excluded(self) -> None:
        # A null label among class-0 rows would normally pull the centroid
        # toward whatever predictor values shared its row. Dropping it first
        # leaves only the genuine class-0 row, so the centroid equals that row.
        features = _df({"x": [0.0, 999.0, 10.0, 12.0]})
        # Index 1 has an unlabelled outcome (NaN). Index 0 is the only true
        # class-0 row; index 2/3 are class 1.
        outcome = _series([0.0, float("nan"), 1.0, 1.0])
        result = fit_class_distance(features, outcome)
        # Class 0 centroid is just the row at index 0: x = 0.
        # (If the NaN row had been included, the mean would have been ~500.)
        np.testing.assert_allclose(result.centroids[:, 0], [0.0])

    def test_classes_unaffected_by_null_outcomes(self) -> None:
        # The NaN row shouldn't introduce a "NaN class".
        features = _df({"x": [0.0, 999.0, 10.0]})
        outcome = _series([0.0, float("nan"), 1.0])
        result = fit_class_distance(features, outcome)
        # Only two real classes, sorted: [0.0, 1.0].
        assert result.classes == [0.0, 1.0]


# ---------------------------------------------------------------------------
# apply_class_distance — output shape & passthrough
# ---------------------------------------------------------------------------


class TestApplyShape:
    """The output has all original columns plus exactly `c` new columns."""

    def test_appends_c_new_columns(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME))
        out = apply_class_distance(features, fit)
        # Original 2 columns + 2 class-distance columns.
        assert out.shape == (4, 4)

    def test_original_columns_preserved_in_place(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME))
        out = apply_class_distance(features, fit)
        # Original column order is preserved; the new columns are appended.
        assert out.columns[:2] == ["x", "y"]

    def test_original_values_unchanged(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME))
        out = apply_class_distance(features, fit)
        # Original values bit-for-bit unchanged.
        np.testing.assert_array_equal(
            out["x"].to_numpy(), np.asarray(SMALL_FEATURES["x"])
        )
        np.testing.assert_array_equal(
            out["y"].to_numpy(), np.asarray(SMALL_FEATURES["y"])
        )

    def test_row_count_matches_input(self) -> None:
        features = _df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        outcome = _series(["a", "a", "b", "b", "a"])
        fit = fit_class_distance(features, outcome)
        out = apply_class_distance(features, fit)
        assert out.shape[0] == 5


# ---------------------------------------------------------------------------
# apply_class_distance — column naming
# ---------------------------------------------------------------------------


class TestApplyColumnNames:
    """New columns follow the `{prefix}_{class}` naming convention."""

    def test_default_prefix(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(["alpha", "alpha", "beta", "beta"]))
        out = apply_class_distance(features, fit)
        assert out.columns[-2:] == ["distance_alpha", "distance_beta"]

    def test_custom_prefix(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(["a", "a", "b", "b"]))
        out = apply_class_distance(features, fit, prefix="dist")
        assert out.columns[-2:] == ["dist_a", "dist_b"]

    def test_integer_class_labels_format_cleanly(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series([3, 3, 7, 7]))
        out = apply_class_distance(features, fit)
        assert out.columns[-2:] == ["distance_3", "distance_7"]


# ---------------------------------------------------------------------------
# apply_class_distance — Euclidean distance values
# ---------------------------------------------------------------------------


class TestApplyDistancesEuclidean:
    """With log_distance=False, raw Euclidean distances match hand calc."""

    def test_row_on_centroid_has_zero_distance(self) -> None:
        # Single row per class -> each row equals its own centroid -> zero.
        features = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(features, _series(["a", "b"]), log_distance=False)
        out = apply_class_distance(features, fit)
        # Row 0 (class "a") is at its own centroid for class "a".
        assert out["distance_a"][0] == pytest.approx(0.0)
        # Row 1 (class "b") is at its own centroid for class "b".
        assert out["distance_b"][1] == pytest.approx(0.0)

    def test_distance_matches_hand_calc(self) -> None:
        # Centroids by hand: class 0 -> (1, 2); class 1 -> (11, 12).
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        out = apply_class_distance(features, fit)
        # Row 0 is (0, 0): distance to (1, 2) = sqrt(1 + 4) = sqrt(5).
        assert out["distance_0"][0] == pytest.approx(math.sqrt(5.0))
        # Row 0 to centroid (11, 12) = sqrt(121 + 144) = sqrt(265).
        assert out["distance_1"][0] == pytest.approx(math.sqrt(265.0))


# ---------------------------------------------------------------------------
# apply_class_distance — Manhattan metric
# ---------------------------------------------------------------------------


class TestMetricManhattan:
    """L1 distance: sum of absolute differences along the predictor axis."""

    def test_row_on_centroid_has_zero_distance(self) -> None:
        features = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(
            features,
            _series(["a", "b"]),
            metric="manhattan",
            log_distance=False,
        )
        out = apply_class_distance(features, fit)
        assert out["distance_a"][0] == pytest.approx(0.0)
        assert out["distance_b"][1] == pytest.approx(0.0)

    def test_distance_matches_hand_calc(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(
            features,
            _series(SMALL_OUTCOME),
            metric="manhattan",
            log_distance=False,
        )
        out = apply_class_distance(features, fit)
        # Row 0 is (0, 0): |0 - 1| + |0 - 2| = 3 (to class 0); |0 - 11| + |0 - 12| = 23 (to class 1).
        assert out["distance_0"][0] == pytest.approx(3.0)
        assert out["distance_1"][0] == pytest.approx(23.0)

    def test_metric_stored_on_fit_object(self) -> None:
        # Sanity: the metric is what we asked for, not silently flipped.
        fit = fit_class_distance(
            _df(SMALL_FEATURES),
            _series(SMALL_OUTCOME),
            metric="manhattan",
        )
        assert fit.metric == "manhattan"


# ---------------------------------------------------------------------------
# apply_class_distance — log_distance and eps
# ---------------------------------------------------------------------------


class TestLogDistance:
    """`log_distance` returns log(raw + eps); `eps` keeps the log finite."""

    def test_log_default_is_true(self) -> None:
        fit = fit_class_distance(_df(SMALL_FEATURES), _series(SMALL_OUTCOME))
        assert fit.log_distance is True

    def test_log_value_matches_log_of_raw_plus_eps(self) -> None:
        features = _df(SMALL_FEATURES)
        raw = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        logged = fit_class_distance(
            features, _series(SMALL_OUTCOME)
        )  # default log=True
        raw_out = apply_class_distance(features, raw)
        log_out = apply_class_distance(features, logged)
        # For every row & class, log_out == log(raw_out + eps).
        for cls in (0, 1):
            col = f"distance_{cls}"
            np.testing.assert_allclose(
                log_out[col].to_numpy(),
                np.log(raw_out[col].to_numpy() + 0.001),
            )

    def test_row_on_centroid_is_finite_with_default_eps(self) -> None:
        # Distance 0 + eps 0.001 -> log(0.001), finite and < 0.
        features = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(features, _series(["a", "b"]))
        out = apply_class_distance(features, fit)
        zero_dist_log = float(out["distance_a"][0])
        assert math.isfinite(zero_dist_log)
        assert zero_dist_log == pytest.approx(math.log(0.001))

    def test_custom_eps_honored(self) -> None:
        features = _df({"x": [3.0, 9.0]})
        fit = fit_class_distance(features, _series(["a", "b"]), eps=0.5)
        out = apply_class_distance(features, fit)
        # Row 0 has raw distance 0 to its own centroid; log(0 + 0.5) = log(0.5).
        assert float(out["distance_a"][0]) == pytest.approx(math.log(0.5))


# ---------------------------------------------------------------------------
# apply_class_distance — missing-value propagation
# ---------------------------------------------------------------------------


class TestApplyMissingValues:
    """A NaN predictor in a row makes that row's distances NaN, others unaffected."""

    def test_nan_predictor_row_becomes_nan(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        # Build a new frame with a NaN in row 1.
        with_nan = _df(
            {"x": [0.0, float("nan"), 10.0, 12.0], "y": [0.0, 4.0, 10.0, 14.0]}
        )
        out = apply_class_distance(with_nan, fit)
        # Row 1's distances should be NaN.
        assert math.isnan(float(out["distance_0"][1]))
        assert math.isnan(float(out["distance_1"][1]))

    def test_other_rows_unaffected_by_nan(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        with_nan = _df(
            {"x": [0.0, float("nan"), 10.0, 12.0], "y": [0.0, 4.0, 10.0, 14.0]}
        )
        out = apply_class_distance(with_nan, fit)
        # Rows 0, 2, 3 must be finite — the NaN in row 1 doesn't propagate.
        for row in (0, 2, 3):
            assert math.isfinite(float(out["distance_0"][row]))
            assert math.isfinite(float(out["distance_1"][row]))


# ---------------------------------------------------------------------------
# Round trip & non-predictor passthrough
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end fit + apply on a small dataset; basic sanity invariants."""

    def test_distances_non_negative_and_finite(self) -> None:
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        out = apply_class_distance(features, fit)
        for col in ("distance_0", "distance_1"):
            arr = out[col].to_numpy()
            assert np.all(np.isfinite(arr))
            assert np.all(arr >= 0.0)

    def test_single_row_class_has_zero_self_distance(self) -> None:
        # When a class has exactly one row, the centroid is that row, so the
        # row's distance to its own class centroid is exactly 0.
        features = _df({"x": [1.0, 5.0, 9.0]})
        fit = fit_class_distance(features, _series(["a", "b", "c"]), log_distance=False)
        out = apply_class_distance(features, fit)
        assert float(out["distance_a"][0]) == pytest.approx(0.0)
        assert float(out["distance_b"][1]) == pytest.approx(0.0)
        assert float(out["distance_c"][2]) == pytest.approx(0.0)

    def test_extra_passthrough_column_kept(self) -> None:
        # An unrelated column on ``data`` should survive apply unchanged.
        features = _df(SMALL_FEATURES)
        fit = fit_class_distance(features, _series(SMALL_OUTCOME), log_distance=False)
        with_extra = _df({**SMALL_FEATURES, "passthrough": [99.0, 99.0, 99.0, 99.0]})
        out = apply_class_distance(with_extra, fit)
        # Extra column survives and keeps its values.
        assert "passthrough" in out.columns
        np.testing.assert_array_equal(
            out["passthrough"].to_numpy(), np.asarray([99.0, 99.0, 99.0, 99.0])
        )
