"""Tests for :mod:`shoez.step_polynomial`.

The functions under test are written against the internal :data:`~shoez.types.Frame`
type (a narwhals ``DataFrame``), so these tests build inputs via
:func:`narwhals.stable.v2.from_native` and inspect outputs as narwhals frames.
We don't parametrize over pandas/polars here — the public contract is
narwhals-in, narwhals-out, and the underlying ops in
:mod:`shoez.step_polynomial` are backend-agnostic by construction.

Ground-truth values for the orthogonal-polynomial output come from R's
:func:`stats::poly`; the matrices below are copied verbatim from
``~/tmp/orth_poly_examples.Rout``.
"""

from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import pytest

from shoez.step_polynomial import PolynomialFit, apply_polynomial, fit_polynomial


def _df(data: dict[str, list[Any]]) -> nw.DataFrame[Any]:
    """Helper: build an internal narwhals Frame from a plain dict.

    Uses pandas under the hood because polars is stricter about constructing
    all-null columns from a plain Python list; the public contract is the
    same either way.
    """
    return nw.from_native(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# fit_polynomial — return shape & ordering
# ---------------------------------------------------------------------------


class TestFitReturnShape:
    """`fit_polynomial` returns an ordered ``dict[str, PolynomialFit]``."""

    def test_returns_dict(self) -> None:
        result = fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=2)
        assert isinstance(result, dict)

    def test_keys_are_strings_values_are_polynomial_fits(self) -> None:
        result = fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=2)
        assert all(isinstance(k, str) for k in result.keys())
        assert all(isinstance(v, PolynomialFit) for v in result.values())

    def test_one_entry_per_input_column(self) -> None:
        """Every column in the input frame appears in the result."""
        frame = _df({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0], "c": [7.0, 8.0, 9.0]})
        result = fit_polynomial(frame, degree=2)
        assert set(result.keys()) == {"a", "b", "c"}

    def test_input_column_order_preserved(self) -> None:
        """The result iterates in the same order as the input frame's columns."""
        frame = _df({"c": [1.0, 2.0, 3.0], "a": [4.0, 5.0, 6.0], "b": [7.0, 8.0, 9.0]})
        result = fit_polynomial(frame, degree=2)
        # `dict` preserves insertion order in Python 3.7+, so list(result)
        # gives the actual iteration order of the result.
        assert list(result.keys()) == ["c", "a", "b"]

    def test_fit_stores_training_mean(self) -> None:
        """The training-set mean is recorded on the fit for centering at apply time."""
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = fit_polynomial(frame, degree=2)
        assert result["x"].mean == pytest.approx(3.0)

    def test_fit_transform_matrix_has_expected_shape(self) -> None:
        """Transform matrix is ``(degree + 1, degree + 1)``."""
        result = fit_polynomial(_df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}), degree=3)
        assert result["x"].transform.shape == (4, 4)


# ---------------------------------------------------------------------------
# fit_polynomial — degree computation (the "min" cap)
# ---------------------------------------------------------------------------


class TestFitDegreeComputation:
    """Effective degree is ``min(degree, n_unique_non_null - 1)``, clamped >= 0."""

    def test_varied_column_gets_full_requested_degree(self) -> None:
        # 5 unique values, degree=3 -> min(3, 5-1) = 3
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        assert fit_polynomial(frame, degree=3)["x"].degree == 3

    def test_degree_capped_by_unique_value_count(self) -> None:
        # 3 unique values, degree=5 -> min(5, 3-1) = 2
        frame = _df({"x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]})
        assert fit_polynomial(frame, degree=5)["x"].degree == 2

    def test_constant_column_gives_degree_zero(self) -> None:
        # 1 unique value, degree=4 -> min(4, 1-1) = 0
        frame = _df({"x": [7.0, 7.0, 7.0]})
        assert fit_polynomial(frame, degree=4)["x"].degree == 0

    def test_all_null_column_gives_degree_zero(self) -> None:
        # 0 unique non-null values, degree=4 -> max(0, min(4, -1)) = 0
        frame = _df({"x": [None, None, None]})
        assert fit_polynomial(frame, degree=4)["x"].degree == 0

    def test_nulls_are_excluded_from_unique_count(self) -> None:
        # 2 unique non-null values plus one null -> min(5, 2-1) = 1
        frame = _df({"x": [1.0, 2.0, None]})
        assert fit_polynomial(frame, degree=5)["x"].degree == 1

    def test_repeated_nulls_do_not_count_as_a_unique_level(self) -> None:
        """Multiple nulls collapse to "no observation" — they don't add a level.

        narwhals' bare `n_unique()` would count null as its own unique value;
        the fitter avoids that by calling `drop_nulls()` first.
        """
        # 3 unique non-null values plus several nulls -> min(5, 3-1) = 2
        frame = _df({"x": [1.0, 2.0, 3.0, None, None, None]})
        assert fit_polynomial(frame, degree=5)["x"].degree == 2

    def test_only_nulls_among_otherwise_constant_values(self) -> None:
        # 1 unique non-null value + nulls -> min(3, 1-1) = 0
        frame = _df({"x": [5.0, None, 5.0, None, 5.0]})
        assert fit_polynomial(frame, degree=3)["x"].degree == 0

    def test_mixed_columns_some_with_nulls(self) -> None:
        """Each column's null-handling is independent of the others."""
        frame = _df(
            {
                "no_nulls": [1.0, 2.0, 3.0, 4.0, 5.0],
                "some_nulls": [1.0, 2.0, None, 4.0, None],  # 3 unique non-null
                "all_nulls": [None, None, None, None, None],  # 0 unique non-null
            }
        )
        result = fit_polynomial(frame, degree=3)
        assert result["no_nulls"].degree == 3  # min(3, 5-1) = 3
        assert result["some_nulls"].degree == 2  # min(3, 3-1) = 2
        assert result["all_nulls"].degree == 0  # max(0, min(3, -1)) = 0

    def test_mixed_columns_each_capped_independently(self) -> None:
        frame = _df(
            {
                "many": [1.0, 2.0, 3.0, 4.0, 5.0],
                "few": [1.0, 1.0, 2.0, 2.0, 2.0],
                "constant": [9.0, 9.0, 9.0, 9.0, 9.0],
            }
        )
        result = fit_polynomial(frame, degree=3)
        # many: min(3, 5-1) = 3
        # few:  min(3, 2-1) = 1
        # constant: min(3, 1-1) = 0
        assert result["many"].degree == 3
        assert result["few"].degree == 1
        assert result["constant"].degree == 0


# ---------------------------------------------------------------------------
# fit_polynomial — input validation on `degree`
# ---------------------------------------------------------------------------


class TestFitDegreeValidation:
    """`degree` must be an `int` strictly greater than zero."""

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=-1)

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=1.5)  # type: ignore[arg-type]

    def test_string_raises(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree="2")  # type: ignore[arg-type]

    def test_bool_true_raises(self) -> None:
        """`True` is technically an `int` subclass, but we reject it explicitly."""
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=True)  # type: ignore[arg-type]

    def test_bool_false_raises(self) -> None:
        with pytest.raises(ValueError, match="degree"):
            fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=False)  # type: ignore[arg-type]

    def test_positive_int_does_not_raise(self) -> None:
        # Sanity: the valid case actually works.
        fit_polynomial(_df({"x": [1.0, 2.0, 3.0]}), degree=1)


# ---------------------------------------------------------------------------
# apply_polynomial — parity with R's poly()
# ---------------------------------------------------------------------------


# Both fixtures below are copied verbatim from ~/tmp/orth_poly_examples.Rout —
# `poly(x, degree)` after `as.numeric` flattening. Used as exact ground truth
# for the orthogonal-polynomial output.

R_POLY_CASE_1_INPUT = [1.0, 2.0, 3.0]
R_POLY_CASE_1_EXPECTED = np.array(
    [
        [-7.071067812e-01, 0.4082482905],
        [-9.681035012e-17, -0.8164965809],
        [7.071067812e-01, 0.4082482905],
    ]
)

R_POLY_CASE_2_INPUT = [
    0.06178627047,
    0.20168193104,
    0.26550866314,
    0.37212389964,
    0.57285336335,
    0.62911404390,
    0.66079779249,
    0.89838968497,
    0.90820778999,
    0.94467526861,
]
R_POLY_CASE_2_EXPECTED = np.array(
    [
        [
            -0.51715435117,
            0.55907132400,
            -0.50468505826,
            0.215522722352,
            -0.12608731662,
            0.04862427607,
            -0.02363920615,
        ],
        [
            -0.36942396079,
            0.07485317045,
            0.30710878096,
            -0.435794813094,
            0.45335732600,
            -0.37737028942,
            0.34839902309,
        ],
        [
            -0.30202267034,
            -0.08947256317,
            0.41868721757,
            -0.165994257276,
            -0.04363404012,
            0.40624746717,
            -0.64150355880,
        ],
        [
            -0.18943654405,
            -0.28486576483,
            0.35379866736,
            0.380162794636,
            -0.52593363568,
            0.06212291808,
            0.46314490079,
        ],
        [
            0.02253459156,
            -0.38421641220,
            -0.17312145689,
            0.318447212338,
            0.06870275164,
            -0.60267677366,
            -0.30616734314,
        ],
        [
            0.08194610058,
            -0.34913079278,
            -0.31274181354,
            -0.007198129523,
            0.18353459722,
            0.06473023653,
            -0.13232771900,
        ],
        [
            0.11540426882,
            -0.31724387569,
            -0.37406843573,
            -0.219651034912,
            0.17304454540,
            0.49644792782,
            0.31937949367,
        ],
        [
            0.36630228014,
            0.20034058550,
            -0.02893866148,
            -0.368332903066,
            -0.38085683246,
            -0.11561583605,
            0.07835355590,
        ],
        [
            0.37667023925,
            0.23230181188,
            0.02916038054,
            -0.223395862404,
            -0.26649554027,
            -0.16292509457,
            -0.16913352550,
        ],
        [
            0.41518004599,
            0.35836251685,
            0.28480037946,
            0.506234270949,
            0.46436814489,
            0.18041516803,
            0.06349437914,
        ],
    ]
)


class TestApplyMatchesRPoly:
    """`apply_polynomial` reproduces R's `poly(x, degree)` element-wise."""

    def test_case_1_degree_2(self) -> None:
        frame = _df({"x": R_POLY_CASE_1_INPUT})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=2))
        actual = out.to_numpy()
        np.testing.assert_allclose(actual, R_POLY_CASE_1_EXPECTED, atol=1e-9)

    def test_case_2_degree_7(self) -> None:
        frame = _df({"x": R_POLY_CASE_2_INPUT})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=7))
        actual = out.to_numpy()
        np.testing.assert_allclose(actual, R_POLY_CASE_2_EXPECTED, atol=1e-9)


# ---------------------------------------------------------------------------
# apply_polynomial — schema / column layout
# ---------------------------------------------------------------------------


class TestApplyColumnLayout:
    """The output frame has the right column names in the right order."""

    def test_polynomial_columns_use_one_indexed_suffix(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0, 4.0]})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=3))
        assert out.columns == ["x_1", "x_2", "x_3"]

    def test_expanded_column_keeps_positional_slot(self) -> None:
        """An input column is replaced in place by its polynomial columns."""
        frame = _df(
            {
                "a": [10.0, 20.0, 30.0, 40.0],
                "b": [1.0, 2.0, 3.0, 4.0],
                "c": [100.0, 200.0, 300.0, 400.0],
            }
        )
        # Only "b" is fitted; "a" and "c" pass through.
        fit = fit_polynomial(frame.select(["b"]), degree=2)
        out = apply_polynomial(frame, fit)
        assert out.columns == ["a", "b_1", "b_2", "c"]

    def test_columns_not_in_fit_result_pass_through_unchanged(self) -> None:
        """Untouched columns retain their values exactly."""
        frame = _df({"a": [10.0, 20.0, 30.0], "b": [1.0, 2.0, 3.0]})
        fit = fit_polynomial(frame.select(["b"]), degree=2)
        out = apply_polynomial(frame, fit)
        # Column `a` should appear unchanged.
        np.testing.assert_array_equal(out["a"].to_numpy(), np.array([10.0, 20.0, 30.0]))

    def test_multiple_columns_expand_simultaneously(self) -> None:
        frame = _df(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [5.0, 6.0, 7.0, 8.0],
            }
        )
        # Different per-column degrees by composing two separate fits.
        fit_a = fit_polynomial(frame.select(["a"]), degree=2)
        fit_b = fit_polynomial(frame.select(["b"]), degree=1)
        out = apply_polynomial(frame, {**fit_a, **fit_b})
        assert out.columns == ["a_1", "a_2", "b_1"]

    def test_fit_result_referring_to_missing_column_is_ignored(self) -> None:
        """Schema drift at apply time is tolerated (matches `_frame.apply_drop`)."""
        # Fit "a" on the real frame; conjure a stale fit for "b" that won't be
        # present in the data we apply to.
        frame = _df({"a": [1.0, 2.0, 3.0]})
        fit_a = fit_polynomial(frame, degree=2)
        fit_b_stale = fit_polynomial(_df({"b": [1.0, 2.0, 3.0, 4.0]}), degree=3)
        out = apply_polynomial(frame, {**fit_a, **fit_b_stale})
        assert out.columns == ["a_1", "a_2"]


# ---------------------------------------------------------------------------
# apply_polynomial — degree-zero handling
# ---------------------------------------------------------------------------


class TestApplyDegreeZero:
    """A degree-0 entry leaves the column alone and emits a warning."""

    def test_constant_column_kept_unchanged(self) -> None:
        frame = _df({"x": [7.0, 7.0, 7.0]})
        with pytest.warns(UserWarning, match="x"):
            out = apply_polynomial(frame, fit_polynomial(frame, degree=2))
        # Column name is preserved, values are unchanged.
        assert out.columns == ["x"]
        np.testing.assert_array_equal(out["x"].to_numpy(), np.array([7.0, 7.0, 7.0]))

    def test_warning_names_offending_column(self) -> None:
        frame = _df({"weird_name": [1.0, 1.0, 1.0]})
        with pytest.warns(UserWarning, match="weird_name"):
            apply_polynomial(frame, fit_polynomial(frame, degree=2))

    def test_round_trip_fit_then_apply_on_constant_column(self) -> None:
        """fit_polynomial on a constant column gives degree 0; apply warns."""
        frame = _df({"x": [5.0, 5.0, 5.0]})
        result = fit_polynomial(frame, degree=3)
        assert result["x"].degree == 0
        with pytest.warns(UserWarning, match="x"):
            out = apply_polynomial(frame, result)
        assert out.columns == ["x"]

    def test_explicit_degree_zero_fit_is_accepted(self) -> None:
        """Callers can also build a PolynomialFit by hand for the degree-0 path."""
        frame = _df({"x": [1.0, 2.0, 3.0]})
        with pytest.warns(UserWarning, match="x"):
            out = apply_polynomial(
                frame, {"x": PolynomialFit(degree=0, mean=float("nan"))}
            )
        assert out.columns == ["x"]


# ---------------------------------------------------------------------------
# apply_polynomial — return type
# ---------------------------------------------------------------------------


class TestApplyReturnType:
    def test_returns_narwhals_dataframe(self) -> None:
        frame = _df({"x": [1.0, 2.0, 3.0]})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=2))
        assert isinstance(out, nw.DataFrame)


# ---------------------------------------------------------------------------
# apply_polynomial — missing-value behavior (per-row pass-through)
# ---------------------------------------------------------------------------


class TestApplyMissingValues:
    """Nulls in the input become NaN in *that row only* of the output.

    ``fit_polynomial`` learns the orthogonal basis on the complete (non-null)
    training rows; ``apply_polynomial`` then evaluates that basis on each row
    of the input series and embeds the result into a full-length, NaN-prefilled
    array. A single missing value therefore affects only its own row.
    """

    def test_row_count_matches_input_even_with_nulls(self) -> None:
        """5 input rows -> 5 output rows, regardless of how many were null."""
        frame = _df({"x": [1.0, 2.0, None, 4.0, 5.0]})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=2))
        assert out.shape == (5, 2)

    def test_null_rows_are_nan_other_rows_are_finite(self) -> None:
        """Only the null-input rows are NaN in the output."""
        frame = _df({"x": [1.0, 2.0, None, 4.0, 5.0]})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=2)).to_numpy()
        # The null row (index 2) is fully NaN.
        assert np.all(np.isnan(out[2]))
        # Every other row is fully finite.
        non_null_rows = np.delete(out, 2, axis=0)
        assert np.all(np.isfinite(non_null_rows))

    def test_non_null_rows_match_r_poly_on_the_complete_subset(self) -> None:
        """Inserting a null shouldn't change the polynomial of the other rows.

        Inputs [1, 2, 3] and [1, 2, None, 3] share the same complete subset
        ([1, 2, 3]) — so the polynomial values for rows 0, 1, and the last
        row should be identical to the original R `poly()` output.
        """
        # The complete subset is the same as R case 1 ([1, 2, 3] at degree 2),
        # so we can reuse the R ground truth.
        frame = _df({"x": [1.0, 2.0, None, 3.0]})
        out = apply_polynomial(frame, fit_polynomial(frame, degree=2)).to_numpy()
        # Non-null rows in the output line up with rows 0, 1, 2 of the
        # original 3-row ground truth.
        np.testing.assert_allclose(out[0], R_POLY_CASE_1_EXPECTED[0], atol=1e-9)
        np.testing.assert_allclose(out[1], R_POLY_CASE_1_EXPECTED[1], atol=1e-9)
        np.testing.assert_allclose(out[3], R_POLY_CASE_1_EXPECTED[2], atol=1e-9)
        # The null row is NaN.
        assert np.all(np.isnan(out[2]))

    def test_multiple_scattered_nulls(self) -> None:
        """Multiple nulls scattered through the column are each masked
        independently and don't affect the other rows' polynomial values."""
        # Complete subset is still [1, 2, 3, 4, 5] — same as a clean fit.
        clean = _df({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
        clean_out = apply_polynomial(clean, fit_polynomial(clean, degree=3)).to_numpy()

        with_nulls = _df({"x": [1.0, None, 2.0, 3.0, None, 4.0, 5.0]})
        out = apply_polynomial(
            with_nulls, fit_polynomial(with_nulls, degree=3)
        ).to_numpy()

        # Non-null rows in `out` correspond to rows 0..4 of `clean_out`.
        non_null_idx = [0, 2, 3, 5, 6]
        for src, dst in enumerate(non_null_idx):
            np.testing.assert_allclose(out[dst], clean_out[src], atol=1e-12)
        # And the two null rows are NaN.
        assert np.all(np.isnan(out[1]))
        assert np.all(np.isnan(out[4]))

    def test_all_null_column_kept_unchanged_and_warns(self) -> None:
        """All-null + effective degree 0 still goes through the degree-0 path."""
        frame = _df({"x": [None, None, None]})
        # `fit_polynomial` returns degree 0 for this column, so the
        # degree-0 branch fires and the column is kept untouched — we never
        # evaluate the basis on an all-null vector.
        with pytest.warns(UserWarning, match="x"):
            out = apply_polynomial(frame, fit_polynomial(frame, degree=3))
        assert out.columns == ["x"]
        assert np.all(np.isnan(out["x"].to_numpy().astype(float)))
