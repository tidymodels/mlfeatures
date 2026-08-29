"""Tests for fit_log, apply_log, and the StepLog guards."""

from __future__ import annotations

import math
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import StepLog
from shoez.step_log import apply_log, fit_log


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_the_column_names(self) -> None:
        assert fit_log(_df(("a", [1.0, 2.0, 3.0]))) == ["a"]

    def test_one_entry_per_column_in_frame_order(self) -> None:
        df = _df(("b", [1.0, 2.0]), ("a", [3.0, 4.0]))
        assert fit_log(df) == ["b", "a"]


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_columns_returns_empty(self) -> None:
        assert fit_log(nw.from_native(pd.DataFrame(), eager_only=True)) == []

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepLog(["a"]).transform(_df(("a", [1.0, 2.0])))

    def test_schema_in_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepLog(["a"]).schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError):
            StepLog(["a"]).schema_out()


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_zero_base_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepLog(["a"], base=0)
        assert str(excinfo.value) == snapshot

    def test_negative_base_raises(self) -> None:
        with pytest.raises(ValueError):
            StepLog(["a"], base=-2.0)

    def test_base_one_raises(self, snapshot: SnapshotAssertion) -> None:
        # log base 1 is undefined: log(1) == 0 would divide every value by zero.
        with pytest.raises(ValueError) as excinfo:
            StepLog(["a"], base=1.0)
        assert str(excinfo.value) == snapshot

    def test_offset_is_not_validated(self) -> None:
        # A negative offset is unusual but legal; the domain warning covers it.
        assert StepLog(["a"], offset=-1.0).offset == -1.0


# ---------------------------------------------------------------------------
# apply_log
# ---------------------------------------------------------------------------


class TestApplyLog:
    def test_preserves_columns_and_order(self) -> None:
        df = _df(("b", [1.0, 2.0, 3.0]), ("a", [10.0, 20.0, 30.0]))
        out = apply_log(df, fit_log(df))
        assert out.columns == ["b", "a"]
        assert out.shape == df.shape

    def test_natural_log_by_default(self) -> None:
        df = _df(("x", [1.0, math.e, math.e**2]))
        out = apply_log(df, fit_log(df))
        assert out["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_unselected_columns_pass_through(self) -> None:
        df = _df(("x", [1.0, math.e]), ("y", [5.0, 6.0]))
        out = apply_log(df, ["x"])
        assert out["x"].to_list() == pytest.approx([0.0, 1.0])
        assert out["y"].to_list() == [5.0, 6.0]

    def test_empty_column_list_is_passthrough(self) -> None:
        df = _df(("x", [1.0, 2.0]), ("y", [4.0, 5.0]))
        out = apply_log(df, [])
        assert out["x"].to_list() == [1.0, 2.0]
        assert out["y"].to_list() == [4.0, 5.0]

    def test_integer_input_yields_float_output(self) -> None:
        df = _df(("a", [1, 10, 100]))
        assert df["a"].dtype.is_integer()  # integer in
        out = apply_log(df, fit_log(df), base=10.0)
        assert out["a"].dtype == nw.Float64  # float out
        assert out["a"].to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_matches_the_closed_form(self) -> None:
        values = [0.5, 1.0, 3.0, 17.0]
        out = apply_log(_df(("x", values)), ["x"], base=3.0, offset=2.0)
        expected = [math.log(v + 2.0) / math.log(3.0) for v in values]
        assert out["x"].to_list() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class TestBase:
    def test_base_ten(self) -> None:
        out = apply_log(_df(("x", [1.0, 10.0, 100.0])), ["x"], base=10.0)
        assert out["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_base_two(self) -> None:
        out = apply_log(_df(("x", [1.0, 2.0, 8.0])), ["x"], base=2.0)
        assert out["x"].to_list() == pytest.approx([0.0, 1.0, 3.0])

    def test_fractional_base_flips_the_sign(self) -> None:
        out = apply_log(_df(("x", [2.0, 8.0])), ["x"], base=0.5)
        assert out["x"].to_list() == pytest.approx([-1.0, -3.0])


# ---------------------------------------------------------------------------
# offset
# ---------------------------------------------------------------------------


class TestOffset:
    def test_offset_shifts_values_before_the_log(self) -> None:
        out = apply_log(_df(("x", [0.0, 1.0, 3.0])), ["x"], base=2.0, offset=1.0)
        assert out["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])

    def test_offset_brings_zeros_into_the_domain(self) -> None:
        df = _df(("x", [0.0, 1.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            columns = fit_log(df, offset=1.0)
        out = apply_log(df, columns, offset=1.0)
        assert all(math.isfinite(v) for v in out["x"].to_list())

    def test_zero_offset_is_the_default(self) -> None:
        df = _df(("x", [1.0, math.e]))
        assert apply_log(df, ["x"])["x"].to_list() == pytest.approx(
            apply_log(df, ["x"], offset=0.0)["x"].to_list()
        )


# ---------------------------------------------------------------------------
# Values outside the logarithm's domain
# ---------------------------------------------------------------------------


class TestNonPositiveValues:
    def test_zero_becomes_negative_infinity(self) -> None:
        with pytest.warns(UserWarning):
            out = apply_log(_df(("x", [0.0, 1.0])), ["x"])
        values = out["x"].to_list()
        assert values[0] == float("-inf")
        assert values[1] == pytest.approx(0.0)

    def test_negative_becomes_nan(self) -> None:
        with pytest.warns(UserWarning):
            out = apply_log(_df(("x", [-1.0, 1.0])), ["x"])
        assert math.isnan(out["x"].to_list()[0])

    def test_fit_warns_and_names_the_columns(self, snapshot: SnapshotAssertion) -> None:
        df = _df(("ok", [1.0, 2.0]), ("bad", [0.0, 2.0]))
        with pytest.warns(UserWarning) as record:
            columns = fit_log(df)
        assert str(record[0].message) == snapshot
        assert columns == ["ok", "bad"]  # kept: -inf is informative output

    def test_fit_warns_once_for_several_columns(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = _df(("a", [0.0, 2.0]), ("b", [-3.0, 2.0]))
        with pytest.warns(UserWarning) as record:
            fit_log(df)
        assert [str(w.message) for w in record] == snapshot

    def test_no_warning_when_all_values_are_in_domain(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_log(_df(("x", [1.0, 2.0, 3.0]))) == ["x"]

    def test_offset_moves_the_domain_boundary(self) -> None:
        # -2 is out of domain at offset 0, in domain at offset 3.
        with pytest.warns(UserWarning):
            fit_log(_df(("x", [-2.0, 1.0])))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_log(_df(("x", [-2.0, 1.0])), offset=3.0) == ["x"]

    def test_negative_offset_can_push_values_out_of_domain(self) -> None:
        with pytest.warns(UserWarning):
            fit_log(_df(("x", [1.0, 2.0])), offset=-1.0)

    def test_transform_warns_on_new_data_that_left_the_domain(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # The training column was strictly positive, so fit() was silent; the
        # new data is not, which is exactly the case the transform-time check
        # exists for.
        train = _df(("x", [1.0, 2.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            columns = fit_log(train)
        with pytest.warns(UserWarning) as record:
            apply_log(_df(("x", [0.0, 2.0])), columns)
        assert str(record[0].message) == snapshot

    def test_transform_silent_when_new_data_is_in_domain(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = apply_log(_df(("x", [1.0, 2.0])), ["x"])
        assert out["x"].to_list()[0] == pytest.approx(0.0)

    def test_numpy_runtime_warnings_are_suppressed(self) -> None:
        # pandas routes through numpy, which raises RuntimeWarning on log(0)
        # and log(negative); the step's own UserWarning is the only signal.
        with pytest.warns(UserWarning) as record:
            apply_log(_df(("x", [0.0, -1.0, 1.0])), ["x"])
        assert [type(w.message) for w in record] == [UserWarning]

    def test_unselected_out_of_domain_column_does_not_warn(self) -> None:
        df = _df(("x", [1.0, 2.0]), ("zero", [0.0, 0.0]))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            apply_log(df, ["x"])


# ---------------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------------


class TestMissingValues:
    def test_nulls_do_not_trigger_the_domain_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_log(_df(("x", [1.0, None, 3.0]))) == ["x"]

    def test_null_rows_stay_missing(self) -> None:
        df = _df(("x", [1.0, None, math.e]))
        out = apply_log(df, fit_log(df))
        values = out["x"].to_list()
        assert values[0] == pytest.approx(0.0)
        assert math.isnan(values[1])
        assert values[2] == pytest.approx(1.0)

    def test_all_null_column_is_kept_without_warning(self) -> None:
        # An all-null column has a missing minimum, so there is nothing to
        # report; the logarithm leaves the nulls alone.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_log(_df(("x", [None, None]))) == ["x"]

    def test_zero_row_column_is_kept_without_warning(self) -> None:
        df = nw.from_native(
            pd.DataFrame({"x": pd.Series([], dtype="float64")}), eager_only=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert fit_log(df) == ["x"]
