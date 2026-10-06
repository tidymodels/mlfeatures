"""Tests for `mlfeatures._spline`, the machinery shared by the two spline steps.

Knot-placement values below were verified against live R output
(`stats::quantile(x, probs = (1:n)/(n+1))`, R's default quantile type 7) —
see the plan for issues tidymodels/shoez#214/#215 for the exact R snippets.
"""

from __future__ import annotations

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import _spline

# ---------------------------------------------------------------------------
# quantile_interior_knots / capped_interior_knots
# ---------------------------------------------------------------------------


class TestQuantileInteriorKnots:
    def test_zero_interior_knots_is_empty(self) -> None:
        knots = _spline.quantile_interior_knots(np.arange(1.0, 11.0), 0)
        assert knots.size == 0

    def test_three_knots_matches_r_quantile(self) -> None:
        # Verified: quantile(1:10, c(0.25, 0.5, 0.75)) == c(3.25, 5.5, 7.75).
        knots = _spline.quantile_interior_knots(np.arange(1.0, 11.0), 3)
        np.testing.assert_allclose(knots, [3.25, 5.5, 7.75])

    def test_two_knots_matches_r_quantile(self) -> None:
        # Verified: quantile(1:10, c(1/3, 2/3)) == c(4, 7).
        knots = _spline.quantile_interior_knots(np.arange(1.0, 11.0), 2)
        np.testing.assert_allclose(knots, [4.0, 7.0])

    def test_one_knot_is_the_median(self) -> None:
        knots = _spline.quantile_interior_knots(np.arange(1.0, 11.0), 1)
        np.testing.assert_allclose(knots, [5.5])


class TestCappedInteriorKnots:
    def test_no_duplicates_returns_full_count(self) -> None:
        knots = _spline.capped_interior_knots(np.arange(1.0, 11.0), 3)
        assert knots.size == 3

    def test_low_cardinality_data_shrinks_knot_count(self) -> None:
        # quantile(c(1,1,1,2), (1:5)/6) collides down to 2 distinct values.
        x = np.array([1.0, 1.0, 1.0, 2.0])
        knots = _spline.capped_interior_knots(x, 5)
        assert knots.size == 2


# ---------------------------------------------------------------------------
# boundary_knots
# ---------------------------------------------------------------------------


class TestBoundaryKnots:
    def test_returns_min_and_max(self) -> None:
        assert _spline.boundary_knots(np.array([3.0, 1.0, 5.0, 2.0])) == (1.0, 5.0)


# ---------------------------------------------------------------------------
# padded_knot_vector / bspline_design_matrix
# ---------------------------------------------------------------------------


class TestPaddedKnotVector:
    def test_boundary_repeated_order_times(self) -> None:
        t = _spline.padded_knot_vector(np.array([3.0, 5.0]), (1.0, 10.0), degree=3)
        # order = degree + 1 = 4 copies of each boundary knot.
        np.testing.assert_array_equal(t[:4], [1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(t[-4:], [10.0, 10.0, 10.0, 10.0])
        np.testing.assert_array_equal(t[4:-4], [3.0, 5.0])


class TestBSplineDesignMatrix:
    def test_shape_is_n_by_interior_plus_order(self) -> None:
        x = np.linspace(1.0, 10.0, 5)
        design = _spline.bspline_design_matrix(
            x, np.array([3.0, 5.0]), (1.0, 10.0), degree=3
        )
        # n_bases = n_interior + degree + 1 = 2 + 4 = 6.
        assert design.shape == (5, 6)

    def test_rows_sum_to_one_inside_boundary(self) -> None:
        """The B-spline partition-of-unity property: rows sum to 1 in-range."""
        x = np.linspace(1.0, 10.0, 9)
        design = _spline.bspline_design_matrix(
            x, np.array([3.0, 5.0, 7.0]), (1.0, 10.0), degree=3
        )
        np.testing.assert_allclose(design.sum(axis=1), np.ones(9), atol=1e-10)


# ---------------------------------------------------------------------------
# check_deg_free / check_degree / check_complete_set
# ---------------------------------------------------------------------------


class TestCheckDegFree:
    def test_valid_does_not_raise(self) -> None:
        _spline.check_deg_free(2, minimum=2)

    def test_below_minimum_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_deg_free(1, minimum=2)
        assert str(excinfo.value) == snapshot

    def test_bool_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_deg_free(True, minimum=2)  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot

    def test_float_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_deg_free(2.5, minimum=2)  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot


class TestCheckDegree:
    def test_valid_does_not_raise(self) -> None:
        _spline.check_degree(0)
        _spline.check_degree(3)

    def test_negative_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_degree(-1)
        assert str(excinfo.value) == snapshot

    def test_bool_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_degree(True)  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot


class TestCheckCompleteSet:
    def test_valid_does_not_raise(self) -> None:
        _spline.check_complete_set(True)
        _spline.check_complete_set(False)

    def test_non_bool_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_complete_set(1)  # type: ignore[arg-type]
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# check_explicit_knots
# ---------------------------------------------------------------------------


class TestCheckExplicitKnots:
    def test_valid_knots_are_sorted(self) -> None:
        result = _spline.check_explicit_knots([7.0, 3.0, 5.0], (1.0, 10.0), column="x")
        np.testing.assert_array_equal(result, [3.0, 5.0, 7.0])

    def test_empty_knots_is_valid(self) -> None:
        result = _spline.check_explicit_knots([], (1.0, 10.0), column="x")
        assert result.size == 0

    def test_duplicate_knots_raise(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_explicit_knots([3.0, 3.0], (1.0, 10.0), column="x")
        assert str(excinfo.value) == snapshot

    def test_knot_at_boundary_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_explicit_knots([1.0, 5.0], (1.0, 10.0), column="x")
        assert str(excinfo.value) == snapshot

    def test_knot_outside_boundary_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_explicit_knots([0.0, 5.0], (1.0, 10.0), column="x")
        assert str(excinfo.value) == snapshot

    def test_error_names_the_column(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _spline.check_explicit_knots([3.0, 3.0], (1.0, 10.0), column="my_col")
        assert str(excinfo.value) == snapshot
