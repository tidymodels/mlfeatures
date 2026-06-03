"""Tests for fit_dummy, apply_dummy, and StepDummy."""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import pytest

from shoez.step_dummy import StepDummy, apply_dummy, fit_dummy
from shoez.types import SchemaInfo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pd(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    """Build a narwhals DataFrame backed by pandas."""
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


def _schema() -> SchemaInfo:
    return SchemaInfo()


def _default_name(col: str, level: str) -> str:
    return f"{col}_{level}"


# ---------------------------------------------------------------------------
# fit_dummy — basic level selection
# ---------------------------------------------------------------------------


class TestFitDummy:
    def test_two_level_string_drops_first(self) -> None:
        df = _pd(("x", ["a", "b"]))
        result = fit_dummy(df)
        assert result == {"x": ["b"]}

    def test_three_level_string_drops_first(self) -> None:
        df = _pd(("x", ["c", "a", "b"]))
        result = fit_dummy(df)
        # levels sorted ascending: a, b, c → drop a → retain b, c
        assert result == {"x": ["b", "c"]}

    def test_levels_sorted_ascending(self) -> None:
        df = _pd(("x", ["banana", "apple", "cherry"]))
        result = fit_dummy(df)
        assert result["x"] == ["banana", "cherry"]

    def test_nulls_excluded_from_levels(self) -> None:
        df = _pd(("x", ["a", None, "b"]))
        result = fit_dummy(df)
        assert result == {"x": ["b"]}

    def test_integer_column(self) -> None:
        df = _pd(("x", [3, 1, 2]))
        result = fit_dummy(df)
        # sorted: 1, 2, 3 → drop 1
        assert result == {"x": [2, 3]}

    def test_boolean_column_drops_false(self) -> None:
        df = _pd(("x", [True, False, True]))
        result = fit_dummy(df)
        # sorted: False, True → drop False
        assert result == {"x": [True]}

    def test_multiple_columns_independent(self) -> None:
        df = _pd(("col1", ["a", "b", "c"]), ("col2", ["x", "y", "x"]))
        result = fit_dummy(df)
        assert result == {"col1": ["b", "c"], "col2": ["y"]}

    def test_return_type(self) -> None:
        df = _pd(("x", ["a", "b"]))
        result = fit_dummy(df)
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_single_level_raises(self) -> None:
        df = _pd(("x", ["a", "a", "a"]))
        with pytest.raises(ValueError, match="fewer than 2"):
            fit_dummy(df)

    def test_all_null_raises(self) -> None:
        df = _pd(("x", [None, None]))
        with pytest.raises(ValueError, match="fewer than 2"):
            fit_dummy(df)

    def test_zero_row_raises(self) -> None:
        df = _pd(("x", []))
        with pytest.raises(ValueError, match="zero-row"):
            fit_dummy(df)


# ---------------------------------------------------------------------------
# apply_dummy — shared transform utility
# ---------------------------------------------------------------------------


class TestApplyDummy:
    def _levels(self) -> dict[str, list[Any]]:
        # Retained levels for "x": "b" and "c" (reference "a" dropped by StepDummy)
        return {"x": ["b", "c"]}

    def _all_known(self) -> dict[str, set[Any]]:
        # Full known set including the reference level "a"
        return {"x": {"a", "b", "c"}}

    def test_known_levels_correct_indicators(self) -> None:
        frame = _pd(("x", ["a", "b", "c"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        result_native = nw.to_native(result)
        assert list(result_native["x_b"]) == [0, 1, 0]
        assert list(result_native["x_c"]) == [0, 0, 1]
        assert "x" not in result_native.columns

    def test_reference_level_produces_all_zeros(self) -> None:
        frame = _pd(("x", ["a", "a"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        native = nw.to_native(result)
        assert list(native["x_b"]) == [0, 0]
        assert list(native["x_c"]) == [0, 0]

    def test_keep_original_cols_true(self) -> None:
        frame = _pd(("x", ["a", "b"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, True, all_known=self._all_known()
        )
        native = nw.to_native(result)
        assert "x" in native.columns
        assert "x_b" in native.columns

    def test_keep_original_cols_false(self) -> None:
        frame = _pd(("x", ["a", "b"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        assert "x" not in nw.to_native(result).columns

    def test_null_rows_produce_null_indicators(self) -> None:
        frame = _pd(("x", ["a", None, "b"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        native = nw.to_native(result)
        # row 1 (None) should be NaN/None in both indicator columns
        assert pd.isna(native["x_b"].iloc[1])
        assert pd.isna(native["x_c"].iloc[1])
        # other rows should be 0 or 1
        assert native["x_b"].iloc[0] == 0
        assert native["x_b"].iloc[2] == 1

    def test_novel_level_produces_null_and_warns(self) -> None:
        frame = _pd(("x", ["a", "b", "UNSEEN"]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_dummy(
                frame,
                self._levels(),
                _default_name,
                False,
                all_known=self._all_known(),
            )
        assert any("not seen" in str(warning.message).lower() for warning in w)
        native = nw.to_native(result)
        # row 2 (UNSEEN) should be null in all indicators
        assert pd.isna(native["x_b"].iloc[2])
        assert pd.isna(native["x_c"].iloc[2])

    def test_novel_and_null_combined(self) -> None:
        frame = _pd(("x", ["a", None, "NOVEL"]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = apply_dummy(
                frame,
                self._levels(),
                _default_name,
                False,
                all_known=self._all_known(),
            )
        assert any("not seen" in str(warning.message).lower() for warning in w)
        native = nw.to_native(result)
        assert pd.isna(native["x_b"].iloc[1])  # null row
        assert pd.isna(native["x_b"].iloc[2])  # novel row

    def test_no_warning_when_all_levels_known(self) -> None:
        frame = _pd(("x", ["a", "b", "c"]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            apply_dummy(
                frame, self._levels(), _default_name, False, all_known=self._all_known()
            )
        assert not any("not seen" in str(warning.message).lower() for warning in w)

    def test_column_name_collision_raises(self) -> None:
        frame = _pd(("x", ["a", "b"]), ("x_b", [10, 20]))
        with pytest.raises(ValueError, match="conflict"):
            apply_dummy(
                frame, self._levels(), _default_name, False, all_known=self._all_known()
            )

    def test_zero_row_frame(self) -> None:
        frame = _pd(("x", []))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        native = nw.to_native(result)
        assert len(native) == 0
        assert "x_b" in native.columns
        assert "x_c" in native.columns

    def test_one_row_frame(self) -> None:
        frame = _pd(("x", ["b"]))
        result = apply_dummy(
            frame, self._levels(), _default_name, False, all_known=self._all_known()
        )
        native = nw.to_native(result)
        assert len(native) == 1
        assert native["x_b"].iloc[0] == 1
        assert native["x_c"].iloc[0] == 0

    def test_custom_name_fn(self) -> None:
        frame = _pd(("x", ["a", "b"]))
        result = apply_dummy(
            frame,
            self._levels(),
            lambda col, lvl: f"enc__{col}__{lvl}",
            False,
            all_known=self._all_known(),
        )
        native = nw.to_native(result)
        assert "enc__x__b" in native.columns
        assert "enc__x__c" in native.columns

    def test_empty_levels_dict_is_no_op(self) -> None:
        frame = _pd(("x", ["a", "b"]))
        result = apply_dummy(frame, {}, _default_name, False)
        assert list(nw.to_native(result).columns) == ["x"]


# ---------------------------------------------------------------------------
# StepDummy — correctness
# ---------------------------------------------------------------------------


class TestStepDummyCorrectness:
    def test_basic_fit_transform(self) -> None:
        df = _pd(("color", ["red", "green", "blue"]))
        step = StepDummy(columns=["color"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        # sorted levels: blue, green, red → drop blue → retain green, red
        assert "color_green" in result.columns
        assert "color_red" in result.columns
        assert "color" not in result.columns
        assert result["color_green"].tolist() == [0, 1, 0]
        assert result["color_red"].tolist() == [1, 0, 0]

    def test_source_col_dropped_by_default(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepDummy(columns=["x"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "x" not in result.columns

    def test_keep_original_cols(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepDummy(columns=["x"], keep_original_cols=True)
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "x" in result.columns
        assert "x_b" in result.columns

    def test_multi_column_input(self) -> None:
        df = _pd(("a", ["x", "y"]), ("b", ["p", "q"]))
        step = StepDummy(columns=["a", "b"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "a_y" in result.columns
        assert "b_q" in result.columns

    def test_name_fn_overrides_default(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepDummy(columns=["x"], name_fn=lambda col, lvl: f"IND_{col}_{lvl}")
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "IND_x_b" in result.columns
        assert "x_b" not in result.columns

    def test_indicator_values_match_expected(self) -> None:
        df = _pd(("g", ["a", "b", "c", "a", "c"]))
        step = StepDummy(columns=["g"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        # reference = a; indicators for b and c
        assert result["g_b"].tolist() == [0, 1, 0, 0, 0]
        assert result["g_c"].tolist() == [0, 0, 1, 0, 1]

    def test_schema_in_after_fit(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepDummy(columns=["x"])
        step.fit(df, _schema())
        si = step.schema_in()
        assert si.columns == ["x"]

    def test_schema_out_after_fit(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepDummy(columns=["x"])
        step.fit(df, _schema())
        so = step.schema_out()
        assert so.columns == ["x_b"]

    def test_polars_backend(self) -> None:
        pl = pytest.importorskip("polars")
        df = nw.from_native(
            pl.DataFrame({"x": pl.Series(["a", "b", "c"])}), eager_only=True
        )
        step = StepDummy(columns=["x"])
        step.fit(df, _schema())
        result = step.transform(df)
        native = nw.to_native(result)
        assert isinstance(native, pl.DataFrame)
        assert "x_b" in native.columns
        assert "x_c" in native.columns


# ---------------------------------------------------------------------------
# StepDummy — guards
# ---------------------------------------------------------------------------


class TestStepDummyGuards:
    def test_transform_before_fit_raises(self) -> None:
        step = StepDummy(columns=["x"])
        df = _pd(("x", ["a"]))
        with pytest.raises(RuntimeError, match="before fit"):
            step.transform(df)

    def test_schema_in_before_fit_raises(self) -> None:
        step = StepDummy(columns=["x"])
        with pytest.raises(RuntimeError, match="before fit"):
            step.schema_in()

    def test_schema_out_before_fit_raises(self) -> None:
        step = StepDummy(columns=["x"])
        with pytest.raises(RuntimeError, match="before fit"):
            step.schema_out()

    def test_single_level_raises_at_fit(self) -> None:
        df = _pd(("x", ["a", "a"]))
        step = StepDummy(columns=["x"])
        with pytest.raises(ValueError, match="fewer than 2"):
            step.fit(df, _schema())

    def test_float_column_rejected_by_accepts_dtypes(self) -> None:
        import narwhals.stable.v2 as nw_

        step = StepDummy(columns=["x"])
        assert nw_.Float64 not in step.accepts_dtypes()
        assert nw_.Float32 not in step.accepts_dtypes()
        assert nw_.Decimal not in step.accepts_dtypes()

    def test_discrete_dtypes_accepted(self) -> None:
        import narwhals.stable.v2 as nw_

        step = StepDummy(columns=["x"])
        accepted = step.accepts_dtypes()
        for dtype in (nw_.String, nw_.Boolean, nw_.Categorical, nw_.Int32, nw_.UInt8):
            assert dtype in accepted

    def test_produces_sparse_false(self) -> None:
        assert StepDummy(columns=["x"]).produces_sparse is False


# ---------------------------------------------------------------------------
# StepDummy — infrastructure
# ---------------------------------------------------------------------------


class TestStepDummyInfrastructure:
    def test_pickle_round_trip(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepDummy(columns=["x"])
        step.fit(df, _schema())
        expected = nw.to_native(step.transform(df))
        restored = pickle.loads(pickle.dumps(step))
        actual = nw.to_native(restored.transform(df))
        pd.testing.assert_frame_equal(expected, actual)

    def test_default_id(self) -> None:
        assert StepDummy(columns=["x"]).id == "dummy"

    def test_custom_id(self) -> None:
        assert StepDummy(columns=["x"], id="my_step").id == "my_step"

    def test_output_role_is_predictor(self) -> None:
        assert StepDummy(columns=["x"]).output_role == "predictor"
