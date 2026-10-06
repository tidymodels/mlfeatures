"""Tests for `fit_spline_monotone` and `apply_spline_monotone`.

Ground-truth matrices are copied verbatim from live R output
(``recipes::step_spline_monotone()``, which wraps ``splines2::iSpline()``) on
``x <- 1:10``, captured while developing this port (see the plan for issues
tidymodels/shoez#214/#215 for the exact R snippets used to generate them).
"""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_spline_monotone import (
    SplineMonotoneFit,
    apply_spline_monotone,
    fit_spline_monotone,
)

X_1_TO_10 = list(range(1, 11))


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_spline_monotone — return shape
# ---------------------------------------------------------------------------


class TestFitReturnShape:
    def test_one_entry_per_column(self) -> None:
        frame = _df({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]})
        result = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        assert set(result) == {"a", "b"}

    def test_column_order_preserved(self) -> None:
        frame = _df({"b": [1.0, 2.0, 3.0, 4.0], "a": [4.0, 5.0, 6.0, 7.0]})
        result = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        assert list(result) == ["b", "a"]

    def test_values_are_spline_monotone_fits(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        result = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        assert isinstance(result["x"], SplineMonotoneFit)


# ---------------------------------------------------------------------------
# fit_spline_monotone — knot count / effective deg_free
# ---------------------------------------------------------------------------


class TestFitKnotCount:
    def test_zero_interior_knots(self) -> None:
        # Verified: deg_free=4, degree=3, complete_set=TRUE -> 0 interior knots.
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)["x"]
        assert fit.internal_knots.size == 0
        assert fit.deg_free == 4

    def test_two_interior_knots(self) -> None:
        # Verified: deg_free=6, degree=3, complete_set=TRUE -> knots at
        # quantile(1:10, c(1/3, 2/3)) == c(4, 7).
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=6, degree=3, complete_set=True)["x"]
        np.testing.assert_allclose(fit.internal_knots, [4.0, 7.0])
        assert fit.deg_free == 6

    def test_complete_set_false_uses_one_fewer_interior_knot(self) -> None:
        # Verified: deg_free=5, degree=3, complete_set=FALSE -> same 2 interior
        # knots as the deg_free=6/complete_set=TRUE case above.
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=5, degree=3, complete_set=False)["x"]
        np.testing.assert_allclose(fit.internal_knots, [4.0, 7.0])
        assert fit.deg_free == 5

    def test_boundary_knots_are_min_and_max(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)["x"]
        assert fit.boundary_knots == (1.0, 10.0)


# ---------------------------------------------------------------------------
# fit_spline_monotone — explicit knots
# ---------------------------------------------------------------------------


class TestFitExplicitKnots:
    def test_explicit_knots_used_verbatim(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(
            frame, deg_free=999, degree=3, complete_set=True, knots=[4.0, 7.0]
        )["x"]
        np.testing.assert_allclose(fit.internal_knots, [4.0, 7.0])
        assert fit.deg_free == 6  # 2 interior + degree(3) + 1 (complete_set=True)

    def test_invalid_explicit_knots_raise(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        with pytest.raises(ValueError) as excinfo:
            fit_spline_monotone(
                frame, deg_free=10, degree=3, complete_set=True, knots=[1.0, 5.0]
            )
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# fit_spline_monotone — validation
# ---------------------------------------------------------------------------


class TestFitValidation:
    def test_deg_free_below_minimum_raises(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError) as excinfo:
            fit_spline_monotone(frame, deg_free=2, degree=3, complete_set=True)
        assert str(excinfo.value) == snapshot

    def test_negative_degree_raises(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError) as excinfo:
            fit_spline_monotone(frame, deg_free=4, degree=-1, complete_set=True)
        assert str(excinfo.value) == snapshot

    def test_deg_free_not_checked_when_knots_given(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        fit_spline_monotone(frame, deg_free=1, degree=3, complete_set=True, knots=[])


# ---------------------------------------------------------------------------
# fit_spline_monotone — degenerate columns
# ---------------------------------------------------------------------------


class TestFitDegenerateColumns:
    def test_constant_column_gets_deg_free_zero(self) -> None:
        frame = _df({"x": [5.0, 5.0, 5.0, 5.0]})
        with pytest.warns(UserWarning):
            fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)[
                "x"
            ]
        assert fit.deg_free == 0

    def test_warning_names_the_column(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"weird_name": [5.0, 5.0, 5.0]})
        with pytest.warns(UserWarning) as record:
            fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        assert str(record[0].message) == snapshot

    def test_no_warning_when_no_degenerate_columns(self) -> None:
        import warnings

        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)


# ---------------------------------------------------------------------------
# apply_spline_monotone — parity with R splines2::iSpline
# ---------------------------------------------------------------------------

# Verified: R `recipes::step_spline_monotone(x, deg_free = 4, degree = 3)`
# (complete_set = TRUE, the default; 0 interior knots) on x <- 1:10.
R_MONOTONE_DF4_DEGREE3 = np.array(
    [
        [0.000000000000000, 0.000000000000000, 0.0000000000000000, 0.000000000000000],
        [0.375704923030026, 0.0635573845450389, 0.0050297210791038, 0.000152415790276],
        [0.634049687547630, 0.2158207590306355, 0.0365797896662094, 0.002438652644414],
        [0.802469135802469, 0.4074074074074073, 0.1111111111111111, 0.012345679012346],
        [0.904740131077580, 0.5999085505258346, 0.2341106538637403, 0.039018442310623],
        [0.960981557689377, 0.7658893461362598, 0.4000914494741655, 0.095259868922420],
        [0.987654320987654, 0.8888888888888890, 0.5925925925925926, 0.197530864197531],
        [0.997561347355586, 0.9634202103337903, 0.7841792409693642, 0.365950312452370],
        [0.999847584209724, 0.9949702789208960, 0.9364426154549609, 0.624295076969974],
        [1.000000000000000, 1.0000000000000000, 1.0000000000000000, 1.000000000000000],
    ]
)

# Verified: R `step_spline_monotone(x, deg_free = 6, degree = 3)` (2 interior
# knots at 4 and 7) on x <- 1:10.
R_MONOTONE_DF6_DEGREE3 = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            0.802469135802469,
            0.233024691358025,
            0.0209190672153635,
            0.000685871056241,
            0.0,
            0.0,
        ],
        [
            0.987654320987654,
            0.617283950617284,
            0.1371742112482854,
            0.010973936899863,
            0.0,
            0.0,
        ],
        [
            1.000000000000000,
            0.875000000000000,
            0.3611111111111112,
            0.055555555555556,
            0.0,
            0.0,
        ],
        [
            1.000000000000000,
            0.975308641975309,
            0.6207133058984911,
            0.172496570644719,
            0.00154320987654321,
            0.0,
        ],
        [
            1.000000000000000,
            0.998456790123457,
            0.8275034293552812,
            0.379286694101509,
            0.02469135802469136,
            0.0,
        ],
        [
            1.000000000000000,
            1.000000000000000,
            0.9444444444444444,
            0.638888888888889,
            0.12500000000000000,
            0.0,
        ],
        [
            1.000000000000000,
            1.000000000000000,
            0.9890260631001372,
            0.862825788751715,
            0.38271604938271603,
            0.0123456790123457,
        ],
        [
            1.000000000000000,
            1.000000000000000,
            0.9993141289437585,
            0.979080932784636,
            0.76697530864197527,
            0.1975308641975309,
        ],
        [
            1.000000000000000,
            1.000000000000000,
            1.0000000000000000,
            1.000000000000000,
            1.00000000000000000,
            1.0000000000000000,
        ],
    ]
)

# Verified: predicting the deg_free=4/degree=3 fit above at x = c(-5, 20).
# R warns "Some 'x' values beyond boundary knots may cause ill-conditioned
# basis functions" and returns the raw polynomial continuation.
R_MONOTONE_EXTRAPOLATION = np.array(
    [
        [-6.716049382716049, 5.62962962962963, -1.77777777777778, 0.197530864197531],
        [-0.524157902758743, 11.05944215820757, -21.95381801554642, 19.862978204541989],
    ]
)


class TestApplyMatchesR:
    def test_zero_interior_knots(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        out = apply_spline_monotone(frame, fit).to_numpy()
        np.testing.assert_allclose(out, R_MONOTONE_DF4_DEGREE3, atol=1e-9)

    def test_two_interior_knots(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=6, degree=3, complete_set=True)
        out = apply_spline_monotone(frame, fit).to_numpy()
        np.testing.assert_allclose(out, R_MONOTONE_DF6_DEGREE3, atol=1e-9)

    def test_out_of_boundary_continues_raw_polynomial(self) -> None:
        train = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(train, deg_free=4, degree=3, complete_set=True)
        new = _df({"x": [-5.0, 20.0]})
        out = apply_spline_monotone(new, fit).to_numpy()
        np.testing.assert_allclose(out, R_MONOTONE_EXTRAPOLATION, atol=1e-6)

    def test_basis_is_non_negative_and_non_decreasing_in_range(self) -> None:
        """Every I-spline basis column is monotone non-decreasing in x, in-range."""
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=6, degree=3, complete_set=True)
        out = apply_spline_monotone(frame, fit).to_numpy()
        assert np.all(out >= -1e-12)
        for col in range(out.shape[1]):
            assert np.all(np.diff(out[:, col]) >= -1e-12)


# ---------------------------------------------------------------------------
# apply_spline_monotone — column layout
# ---------------------------------------------------------------------------


class TestApplyColumnLayout:
    def test_basis_columns_use_one_indexed_suffix(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        out = apply_spline_monotone(
            frame,
            fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True),
        )
        assert out.columns == ["x_1", "x_2", "x_3", "x_4"]

    def test_expanded_column_keeps_positional_slot(self) -> None:
        frame = _df(
            {
                "a": [float(v) for v in X_1_TO_10],
                "b": [float(v) for v in X_1_TO_10],
                "c": [float(v) for v in X_1_TO_10],
            }
        )
        fit = fit_spline_monotone(
            frame.select(["b"]), deg_free=4, degree=3, complete_set=True
        )
        out = apply_spline_monotone(frame, fit)
        assert out.columns == ["a", "b_1", "b_2", "b_3", "b_4", "c"]

    def test_keep_original_cols_retains_source_before_its_terms(self) -> None:
        frame = _df(
            {
                "a": [float(v) for v in X_1_TO_10],
                "b": [float(v) for v in X_1_TO_10],
            }
        )
        fit = fit_spline_monotone(
            frame.select(["b"]), deg_free=4, degree=3, complete_set=True
        )
        out = apply_spline_monotone(frame, fit, keep_original_cols=True)
        assert out.columns == ["a", "b", "b_1", "b_2", "b_3", "b_4"]

    def test_unselected_columns_pass_through(self) -> None:
        frame = _df({"a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]})
        fit = fit_spline_monotone(
            frame.select(["b"]), deg_free=4, degree=3, complete_set=True
        )
        out = apply_spline_monotone(frame, fit)
        np.testing.assert_array_equal(out["a"].to_numpy(), [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# apply_spline_monotone — degenerate (deg_free == 0) handling
# ---------------------------------------------------------------------------


class TestApplyDegenerate:
    def test_constant_column_kept_unchanged(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [7.0, 7.0, 7.0, 7.0]})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        with pytest.warns(UserWarning) as record:
            out = apply_spline_monotone(frame, fit)
        assert str(record[0].message) == snapshot
        assert out.columns == ["x"]
        np.testing.assert_array_equal(out["x"].to_numpy(), [7.0, 7.0, 7.0, 7.0])

    def test_explicit_degenerate_fit_is_accepted(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0]})
        with pytest.warns(UserWarning):
            out = apply_spline_monotone(
                frame,
                {
                    "x": SplineMonotoneFit(
                        internal_knots=np.array([]),
                        boundary_knots=(float("nan"), float("nan")),
                        degree=3,
                        complete_set=True,
                        deg_free=0,
                    )
                },
            )
        assert out.columns == ["x"]


# ---------------------------------------------------------------------------
# apply_spline_monotone — return type / missing values
# ---------------------------------------------------------------------------


class TestApplyReturnType:
    def test_returns_narwhals_dataframe(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        out = apply_spline_monotone(frame, fit)
        assert isinstance(out, nw.DataFrame)

    def test_integer_input_yields_float_output(self) -> None:
        frame = _df({"x": X_1_TO_10})
        fit = fit_spline_monotone(frame, deg_free=4, degree=3, complete_set=True)
        out = apply_spline_monotone(frame, fit)
        for col in out.columns:
            assert out[col].dtype in (nw.Float32, nw.Float64)

    def test_null_row_becomes_nan_others_unaffected(self) -> None:
        clean = _df({"x": [float(v) for v in X_1_TO_10]})
        clean_out = apply_spline_monotone(
            clean,
            fit_spline_monotone(clean, deg_free=4, degree=3, complete_set=True),
        ).to_numpy()

        with_null = _df({"x": [float(v) for v in X_1_TO_10] + [None]})
        fit = fit_spline_monotone(with_null, deg_free=4, degree=3, complete_set=True)
        out = apply_spline_monotone(with_null, fit).to_numpy()

        np.testing.assert_allclose(out[:10], clean_out, atol=1e-9)
        assert np.all(np.isnan(out[10]))
