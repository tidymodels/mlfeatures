"""Tests for `fit_spline_natural` and `apply_spline_natural`.

Ground-truth matrices are copied verbatim from live R output
(``recipes::step_spline_natural()``, which wraps ``splines2::naturalSpline()``)
on ``x <- 1:10``, captured while developing this port (see the plan for issues
tidymodels/shoez#214/#215 for the exact R snippets used to generate them).
"""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_spline_natural import (
    SplineNaturalFit,
    apply_spline_natural,
    fit_spline_natural,
)

X_1_TO_10 = list(range(1, 11))


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_spline_natural — return shape
# ---------------------------------------------------------------------------


class TestFitReturnShape:
    def test_one_entry_per_column(self) -> None:
        frame = _df({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 5.0, 6.0, 7.0]})
        result = fit_spline_natural(frame, deg_free=2, complete_set=False)
        assert set(result) == {"a", "b"}

    def test_column_order_preserved(self) -> None:
        frame = _df({"b": [1.0, 2.0, 3.0, 4.0], "a": [4.0, 5.0, 6.0, 7.0]})
        result = fit_spline_natural(frame, deg_free=2, complete_set=False)
        assert list(result) == ["b", "a"]

    def test_values_are_spline_natural_fits(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        result = fit_spline_natural(frame, deg_free=2, complete_set=False)
        assert isinstance(result["x"], SplineNaturalFit)


# ---------------------------------------------------------------------------
# fit_spline_natural — knot count / effective deg_free
# ---------------------------------------------------------------------------


class TestFitKnotCount:
    def test_complete_set_false_uses_deg_free_minus_one_interior_knots(self) -> None:
        # Verified: deg_free=4, complete_set=FALSE -> 3 interior knots at
        # quantile(1:10, c(.25, .5, .75)) == c(3.25, 5.5, 7.75).
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=4, complete_set=False)["x"]
        np.testing.assert_allclose(fit.internal_knots, [3.25, 5.5, 7.75])
        assert fit.deg_free == 4

    def test_complete_set_true_uses_deg_free_minus_two_interior_knots(self) -> None:
        # Verified: deg_free=4, complete_set=TRUE -> 2 interior knots at
        # quantile(1:10, c(1/3, 2/3)) == c(4, 7).
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=4, complete_set=True)["x"]
        np.testing.assert_allclose(fit.internal_knots, [4.0, 7.0])
        assert fit.deg_free == 4

    def test_zero_interior_knots(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=2, complete_set=True)["x"]
        assert fit.internal_knots.size == 0
        assert fit.deg_free == 2

    def test_boundary_knots_are_min_and_max(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=3, complete_set=False)["x"]
        assert fit.boundary_knots == (1.0, 10.0)


# ---------------------------------------------------------------------------
# fit_spline_natural — explicit knots
# ---------------------------------------------------------------------------


class TestFitExplicitKnots:
    def test_explicit_knots_used_verbatim(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(
            frame, deg_free=10, complete_set=False, knots=[4.0, 7.0]
        )["x"]
        np.testing.assert_allclose(fit.internal_knots, [4.0, 7.0])
        assert fit.deg_free == 3  # 2 interior + 1 (complete_set=False)

    def test_deg_free_ignored_when_knots_given(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=999, complete_set=True, knots=[5.5])[
            "x"
        ]
        assert fit.deg_free == 3  # 1 interior + 2 (complete_set=True)

    def test_invalid_explicit_knots_raise(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        with pytest.raises(ValueError) as excinfo:
            fit_spline_natural(frame, deg_free=10, complete_set=False, knots=[1.0, 5.0])
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# fit_spline_natural — validation
# ---------------------------------------------------------------------------


class TestFitValidation:
    def test_deg_free_below_two_raises(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        with pytest.raises(ValueError) as excinfo:
            fit_spline_natural(frame, deg_free=1, complete_set=False)
        assert str(excinfo.value) == snapshot

    def test_deg_free_not_checked_when_knots_given(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        # deg_free=1 would normally raise, but it's ignored when knots is given.
        fit_spline_natural(frame, deg_free=1, complete_set=False, knots=[])


# ---------------------------------------------------------------------------
# fit_spline_natural — degenerate columns
# ---------------------------------------------------------------------------


class TestFitDegenerateColumns:
    def test_constant_column_gets_deg_free_zero(self) -> None:
        frame = _df({"x": [5.0, 5.0, 5.0, 5.0]})
        with pytest.warns(UserWarning):
            fit = fit_spline_natural(frame, deg_free=4, complete_set=False)["x"]
        assert fit.deg_free == 0

    def test_warning_names_the_column(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"weird_name": [5.0, 5.0, 5.0]})
        with pytest.warns(UserWarning) as record:
            fit_spline_natural(frame, deg_free=4, complete_set=False)
        assert str(record[0].message) == snapshot

    def test_no_warning_when_no_degenerate_columns(self) -> None:
        import warnings

        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fit_spline_natural(frame, deg_free=2, complete_set=False)

    def test_all_null_column_is_degenerate(self) -> None:
        frame = _df({"x": [None, None, None, None]})
        with pytest.warns(UserWarning):
            fit = fit_spline_natural(frame, deg_free=4, complete_set=False)["x"]
        assert fit.deg_free == 0


# ---------------------------------------------------------------------------
# apply_spline_natural — parity with R splines2::naturalSpline
# ---------------------------------------------------------------------------

# Verified: R `recipes::step_spline_natural(x, deg_free = 4)` (complete_set =
# FALSE, the default) on x <- 1:10.
R_NATURAL_DF4_COMPLETE_FALSE = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.31138545953360763, 0.0146319158664838, 0.0, 0.0],
        [0.49108367626886135, 0.1170553269318701, 0.0, 0.0],
        [
            0.43055555555555558,
            0.3703703703703703,
            0.00462962962962963,
            0.00205761316872428,
        ],
        [
            0.22273662551440329,
            0.6227709190672154,
            0.05881344307270234,
            0.02613930803231215,
        ],
        [
            0.05881344307270234,
            0.6227709190672154,
            0.22273662551440329,
            0.09960371894528272,
        ],
        [
            0.00462962962962963,
            0.3703703703703703,
            0.43055555555555558,
            0.20781893004115226,
        ],
        [0.0, 0.1170553269318701, 0.49108367626886135, 0.29431489102270991],
        [0.0, 0.0146319158664838, 0.31138545953360763, 0.32845602804450535],
        [0.0, 0.0, 0.0, 0.33333333333333331],
    ]
)

# Verified: R `step_spline_natural(x, deg_free = 2, complete_set = TRUE)`
# (0 interior knots) on x <- 1:10.
R_NATURAL_DF2_COMPLETE_TRUE = np.array(
    [
        [0.5, 0.0],
        [0.4444444444444444, 0.0555555555555555],
        [0.3888888888888889, 0.1111111111111111],
        [0.3333333333333333, 0.1666666666666667],
        [0.2777777777777778, 0.2222222222222222],
        [0.2222222222222222, 0.2777777777777778],
        [0.1666666666666667, 0.3333333333333333],
        [0.1111111111111111, 0.3888888888888889],
        [0.0555555555555555, 0.4444444444444444],
        [0.0, 0.5],
    ]
)

# Verified: R `step_spline_natural(x, deg_free = 3, complete_set = TRUE)`
# (1 interior knot, at the median 5.5) on x <- 1:10.
R_NATURAL_DF3_COMPLETE_TRUE = np.array(
    [
        [0.6, 0.0, 0.0],
        [0.46776406035665297, 0.131138545953361, 0.00109739368998628],
        [0.34211248285322360, 0.249108367626886, 0.00877914951989026],
        [0.22962962962962963, 0.340740740740741, 0.02962962962962963],
        [0.13689986282578873, 0.392866941015089, 0.07023319615912209],
        [0.07023319615912209, 0.392866941015089, 0.13689986282578873],
        [0.02962962962962963, 0.340740740740741, 0.22962962962962963],
        [0.00877914951989026, 0.249108367626886, 0.34211248285322360],
        [0.00109739368998628, 0.131138545953361, 0.46776406035665297],
        [0.0, 0.0, 0.6],
    ]
)

# Verified: predicting the above (1-interior-knot) fit at x = c(-5, 20).
R_NATURAL_EXTRAPOLATION = np.array(
    [
        [1.4, -0.8, 0.0],
        [0.0, -1.33333333333333, 1.93333333333333],
    ]
)


class TestApplyMatchesR:
    def test_three_interior_knots_complete_set_false(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=4, complete_set=False)
        out = apply_spline_natural(frame, fit).to_numpy()
        np.testing.assert_allclose(out, R_NATURAL_DF4_COMPLETE_FALSE, atol=1e-9)

    def test_zero_interior_knots_complete_set_true(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=2, complete_set=True)
        out = apply_spline_natural(frame, fit).to_numpy()
        np.testing.assert_allclose(out, R_NATURAL_DF2_COMPLETE_TRUE, atol=1e-9)

    def test_one_interior_knot_complete_set_true(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=3, complete_set=True)
        out = apply_spline_natural(frame, fit).to_numpy()
        np.testing.assert_allclose(out, R_NATURAL_DF3_COMPLETE_TRUE, atol=1e-9)

    def test_out_of_boundary_extrapolates_linearly(self) -> None:
        train = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(train, deg_free=3, complete_set=True)
        new = _df({"x": [-5.0, 20.0]})
        out = apply_spline_natural(new, fit).to_numpy()
        np.testing.assert_allclose(out, R_NATURAL_EXTRAPOLATION, atol=1e-7)


# ---------------------------------------------------------------------------
# apply_spline_natural — column layout
# ---------------------------------------------------------------------------


class TestApplyColumnLayout:
    def test_basis_columns_use_one_indexed_suffix(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        out = apply_spline_natural(
            frame, fit_spline_natural(frame, deg_free=3, complete_set=False)
        )
        assert out.columns == ["x_1", "x_2", "x_3"]

    def test_expanded_column_keeps_positional_slot(self) -> None:
        frame = _df(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [float(v) for v in X_1_TO_10[:4]],
                "c": [10.0, 20.0, 30.0, 40.0],
            }
        )
        fit = fit_spline_natural(frame.select(["b"]), deg_free=2, complete_set=False)
        out = apply_spline_natural(frame, fit)
        assert out.columns == ["a", "b_1", "b_2", "c"]

    def test_keep_original_cols_retains_source_before_its_terms(self) -> None:
        frame = _df({"a": [1.0, 2.0, 3.0, 4.0], "b": [float(v) for v in X_1_TO_10[:4]]})
        fit = fit_spline_natural(frame.select(["b"]), deg_free=2, complete_set=False)
        out = apply_spline_natural(frame, fit, keep_original_cols=True)
        assert out.columns == ["a", "b", "b_1", "b_2"]

    def test_unselected_columns_pass_through(self) -> None:
        frame = _df({"a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]})
        fit = fit_spline_natural(frame.select(["b"]), deg_free=2, complete_set=False)
        out = apply_spline_natural(frame, fit)
        np.testing.assert_array_equal(out["a"].to_numpy(), [10.0, 20.0, 30.0])


# ---------------------------------------------------------------------------
# apply_spline_natural — degenerate (deg_free == 0) handling
# ---------------------------------------------------------------------------


class TestApplyDegenerate:
    def test_constant_column_kept_unchanged(self, snapshot: SnapshotAssertion) -> None:
        frame = _df({"x": [7.0, 7.0, 7.0, 7.0]})
        fit = fit_spline_natural(frame, deg_free=4, complete_set=False)
        with pytest.warns(UserWarning) as record:
            out = apply_spline_natural(frame, fit)
        assert str(record[0].message) == snapshot
        assert out.columns == ["x"]
        np.testing.assert_array_equal(out["x"].to_numpy(), [7.0, 7.0, 7.0, 7.0])

    def test_explicit_degenerate_fit_is_accepted(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0]})
        with pytest.warns(UserWarning):
            out = apply_spline_natural(
                frame,
                {
                    "x": SplineNaturalFit(
                        internal_knots=np.array([]),
                        boundary_knots=(float("nan"), float("nan")),
                        complete_set=False,
                        deg_free=0,
                    )
                },
            )
        assert out.columns == ["x"]


# ---------------------------------------------------------------------------
# apply_spline_natural — return type / missing values
# ---------------------------------------------------------------------------


class TestApplyReturnType:
    def test_returns_narwhals_dataframe(self) -> None:
        frame = _df({"x": [float(v) for v in X_1_TO_10]})
        fit = fit_spline_natural(frame, deg_free=3, complete_set=False)
        out = apply_spline_natural(frame, fit)
        assert isinstance(out, nw.DataFrame)

    def test_integer_input_yields_float_output(self) -> None:
        frame = _df({"x": X_1_TO_10})
        fit = fit_spline_natural(frame, deg_free=3, complete_set=False)
        out = apply_spline_natural(frame, fit)
        for col in out.columns:
            assert out[col].dtype in (nw.Float32, nw.Float64)

    def test_null_row_becomes_nan_others_unaffected(self) -> None:
        clean = _df({"x": [float(v) for v in X_1_TO_10]})
        clean_out = apply_spline_natural(
            clean, fit_spline_natural(clean, deg_free=3, complete_set=False)
        ).to_numpy()

        with_null = _df({"x": [float(v) for v in X_1_TO_10] + [None]})
        fit = fit_spline_natural(with_null, deg_free=3, complete_set=False)
        out = apply_spline_natural(with_null, fit).to_numpy()

        np.testing.assert_allclose(out[:10], clean_out, atol=1e-9)
        assert np.all(np.isnan(out[10]))
