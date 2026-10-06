"""Tests for fit_one_hot and StepOneHot."""

from __future__ import annotations

import pickle
import warnings
from typing import Any

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures.step_one_hot import StepOneHot, fit_one_hot
from mlfeatures.types import Roles, SchemaInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pd(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    """Build a narwhals DataFrame backed by pandas."""
    return nw.from_native(pd.DataFrame(dict(cols)), eager_only=True)


def _schema() -> SchemaInfo:
    return SchemaInfo()


# ---------------------------------------------------------------------------
# fit_one_hot — level selection (all levels retained)
# ---------------------------------------------------------------------------


class TestFitOneHot:
    def test_two_level_string_retains_both(self) -> None:
        df = _pd(("x", ["a", "b"]))
        result = fit_one_hot(df)
        assert result == {"x": ["a", "b"]}

    def test_three_level_string_retains_all(self) -> None:
        df = _pd(("x", ["c", "a", "b"]))
        result = fit_one_hot(df)
        assert result == {"x": ["a", "b", "c"]}

    def test_levels_sorted_ascending(self) -> None:
        df = _pd(("x", ["banana", "apple", "cherry"]))
        result = fit_one_hot(df)
        assert result["x"] == ["apple", "banana", "cherry"]

    def test_nulls_excluded_from_levels(self) -> None:
        df = _pd(("x", ["a", None, "b"]))
        result = fit_one_hot(df)
        assert result == {"x": ["a", "b"]}

    def test_integer_column_retains_all(self) -> None:
        df = _pd(("x", [3, 1, 2]))
        result = fit_one_hot(df)
        assert result == {"x": [1, 2, 3]}

    def test_boolean_column_retains_both(self) -> None:
        df = _pd(("x", [True, False, True]))
        result = fit_one_hot(df)
        assert result == {"x": [False, True]}

    def test_multiple_columns_independent(self) -> None:
        df = _pd(("col1", ["a", "b", "c"]), ("col2", ["x", "y", "x"]))
        result = fit_one_hot(df)
        assert result == {"col1": ["a", "b", "c"], "col2": ["x", "y"]}

    def test_return_type(self) -> None:
        df = _pd(("x", ["a", "b"]))
        result = fit_one_hot(df)
        assert isinstance(result, dict)
        assert all(isinstance(v, list) for v in result.values())

    def test_single_level_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _pd(("x", ["a", "a", "a"]))
        with pytest.raises(ValueError) as excinfo:
            fit_one_hot(df)
        assert str(excinfo.value) == snapshot

    def test_all_null_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _pd(("x", [None, None]))
        with pytest.raises(ValueError) as excinfo:
            fit_one_hot(df)
        assert str(excinfo.value) == snapshot

    def test_zero_row_raises(self, snapshot: SnapshotAssertion) -> None:
        df = _pd(("x", []))
        with pytest.raises(ValueError) as excinfo:
            fit_one_hot(df)
        assert str(excinfo.value) == snapshot


# ---------------------------------------------------------------------------
# StepOneHot — correctness. Null/unseen-level handling is re-tested here
# because StepOneHot drives apply_dummy via the all_known=None path, which
# differs from the explicit all_known set that StepDummy passes.
# ---------------------------------------------------------------------------


class TestStepOneHotCorrectness:
    def test_basic_fit_transform_k_columns(self) -> None:
        df = _pd(("color", ["red", "green", "blue"]))
        step = StepOneHot(columns=["color"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        # All three levels retained: blue, green, red
        assert "color_blue" in result.columns
        assert "color_green" in result.columns
        assert "color_red" in result.columns
        assert "color" not in result.columns

    def test_indicator_values_all_levels(self) -> None:
        df = _pd(("g", ["a", "b", "c", "a"]))
        step = StepOneHot(columns=["g"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert result["g_a"].tolist() == [1, 0, 0, 1]
        assert result["g_b"].tolist() == [0, 1, 0, 0]
        assert result["g_c"].tolist() == [0, 0, 1, 0]

    def test_one_more_column_than_dummy(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        dummy_step = __import__("mlfeatures.step_dummy", fromlist=["StepDummy"]).StepDummy(
            columns=["x"]
        )
        ohe_step = StepOneHot(columns=["x"])
        dummy_step.fit(df, _schema())
        ohe_step.fit(df, _schema())
        dummy_cols = dummy_step.schema_out().columns
        ohe_cols = ohe_step.schema_out().columns
        assert len(ohe_cols) == len(dummy_cols) + 1

    def test_keep_original_cols(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepOneHot(columns=["x"], keep_original_cols=True)
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "x" in result.columns
        assert "x_a" in result.columns
        assert "x_b" in result.columns

    def test_multi_column_input(self) -> None:
        df = _pd(("a", ["x", "y"]), ("b", ["p", "q"]))
        step = StepOneHot(columns=["a", "b"])
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "a_x" in result.columns
        assert "a_y" in result.columns
        assert "b_p" in result.columns
        assert "b_q" in result.columns

    def test_name_fn_overrides_default(self) -> None:
        df = _pd(("x", ["a", "b"]))
        step = StepOneHot(columns=["x"], name_fn=lambda col, lvl: f"OH_{col}_{lvl}")
        step.fit(df, _schema())
        result = nw.to_native(step.transform(df))
        assert "OH_x_a" in result.columns
        assert "OH_x_b" in result.columns

    def test_schema_out_lists_all_k_columns(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        assert step.schema_out().columns == ["x_a", "x_b", "x_c"]

    def test_polars_backend(self) -> None:
        pl = pytest.importorskip("polars")
        df = nw.from_native(
            pl.DataFrame({"x": pl.Series(["a", "b", "c"])}), eager_only=True
        )
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        result = step.transform(df)
        native = nw.to_native(result)
        assert isinstance(native, pl.DataFrame)
        assert "x_a" in native.columns
        assert "x_b" in native.columns
        assert "x_c" in native.columns

    def test_null_rows_produce_null_indicators(self) -> None:
        # Exercises the all_known=None path through StepOneHot (where retained
        # == all known levels): null source rows must be null in every indicator.
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        new = _pd(("x", ["a", None, "b"]))
        result = nw.to_native(step.transform(new))
        # row 1 (None) is null in all indicator columns
        assert pd.isna(result["x_a"].iloc[1])
        assert pd.isna(result["x_b"].iloc[1])
        assert pd.isna(result["x_c"].iloc[1])
        # known rows are 0/1
        assert result["x_a"].iloc[0] == 1
        assert result["x_b"].iloc[2] == 1

    def test_unseen_level_produces_null_and_warns(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        new = _pd(("x", ["a", "b", "UNSEEN"]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = nw.to_native(step.transform(new))
        assert any("not seen" in str(warning.message).lower() for warning in w)
        # row 2 (UNSEEN) is null in all indicators
        assert pd.isna(result["x_a"].iloc[2])
        assert pd.isna(result["x_b"].iloc[2])
        assert pd.isna(result["x_c"].iloc[2])

    def test_unseen_and_null_combined(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        new = _pd(("x", ["a", None, "NOVEL"]))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = nw.to_native(step.transform(new))
        assert any("not seen" in str(warning.message).lower() for warning in w)
        assert pd.isna(result["x_a"].iloc[1])  # null row
        assert pd.isna(result["x_a"].iloc[2])  # unseen row

    def test_no_warning_when_all_levels_known(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            step.transform(_pd(("x", ["a", "b", "c"])))
        assert not any("not seen" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# StepOneHot — guards
# ---------------------------------------------------------------------------


class TestStepOneHotGuards:
    def test_transform_before_fit_raises(self, snapshot: SnapshotAssertion) -> None:
        step = StepOneHot(columns=["x"])
        df = _pd(("x", ["a"]))
        with pytest.raises(RuntimeError) as excinfo:
            step.transform(df)
        assert str(excinfo.value) == snapshot

    def test_schema_in_after_fit(self) -> None:
        df = _pd(("a", ["x", "y"]), ("b", ["p", "q"]))
        step = StepOneHot(columns=["a", "b"])
        step.fit(df, _schema())
        si = step.schema_in()
        # records the consumed source columns and their input dtypes
        assert si.columns == ["a", "b"]
        assert set(si.dtypes) == {"a", "b"}

    def test_schema_in_before_fit_raises(self, snapshot: SnapshotAssertion) -> None:
        step = StepOneHot(columns=["x"])
        with pytest.raises(RuntimeError) as excinfo:
            step.schema_in()
        assert str(excinfo.value) == snapshot

    def test_schema_out_before_fit_raises(self, snapshot: SnapshotAssertion) -> None:
        step = StepOneHot(columns=["x"])
        with pytest.raises(RuntimeError) as excinfo:
            step.schema_out()
        assert str(excinfo.value) == snapshot

    def test_single_level_raises_at_fit(self, snapshot: SnapshotAssertion) -> None:
        df = _pd(("x", ["a", "a"]))
        step = StepOneHot(columns=["x"])
        with pytest.raises(ValueError) as excinfo:
            step.fit(df, _schema())
        assert str(excinfo.value) == snapshot

    def test_float_column_not_in_accepts_dtypes(self) -> None:
        step = StepOneHot(columns=["x"])
        assert nw.Float64 not in step.accepts_dtypes()
        assert nw.Float32 not in step.accepts_dtypes()
        assert nw.Decimal not in step.accepts_dtypes()

    def test_discrete_dtypes_accepted(self) -> None:
        step = StepOneHot(columns=["x"])
        accepted = step.accepts_dtypes()
        for dtype in (nw.String, nw.Boolean, nw.Categorical, nw.Int32, nw.UInt8):
            assert dtype in accepted

    def test_produces_sparse_false(self) -> None:
        assert StepOneHot(columns=["x"]).produces_sparse is False


# ---------------------------------------------------------------------------
# StepOneHot — infrastructure
# ---------------------------------------------------------------------------


class TestStepOneHotInfrastructure:
    def test_pickle_round_trip(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"])
        step.fit(df, _schema())
        expected = nw.to_native(step.transform(df))
        restored = pickle.loads(pickle.dumps(step))
        actual = nw.to_native(restored.transform(df))
        pd.testing.assert_frame_equal(expected, actual)

    def test_default_id(self) -> None:
        assert StepOneHot(columns=["x"]).id == "one_hot"

    def test_custom_id(self) -> None:
        assert StepOneHot(columns=["x"], id="my_ohe").id == "my_ohe"

    def test_output_role_is_feature(self) -> None:
        assert StepOneHot(columns=["x"]).output_role == "feature"


# ---------------------------------------------------------------------------
# Indicator dtype: Int8 when clean, Float64 when null/unseen rows appear
# ---------------------------------------------------------------------------


class TestIndicatorDtype:
    """Indicator columns must be numeric so downstream numeric steps accept them."""

    def test_clean_indicators_are_int8_numeric_pandas(self) -> None:
        df = _pd(("cat", ["a", "b", "c", "a", "b", "c"]))
        out = nw.from_native(
            StepOneHot(columns=["cat"]).fit(df, _schema()).transform(df)
        )
        for col in out.columns:
            assert out.schema[col] == nw.Int8
            assert out.schema[col].is_numeric()

    def test_clean_indicators_are_int8_numeric_polars(self) -> None:
        df = nw.from_native(
            pl.DataFrame({"cat": ["a", "b", "c", "a", "b", "c"]}), eager_only=True
        )
        out = StepOneHot(columns=["cat"]).fit(df, _schema()).transform(df)
        for col in out.columns:
            assert out.schema[col] == nw.Int8
            assert out.schema[col].is_numeric()

    def test_null_source_rows_force_float64_numeric(self) -> None:
        df = _pd(("cat", ["a", "b", None]))
        out = nw.from_native(
            StepOneHot(columns=["cat"]).fit(df, _schema()).transform(df)
        )
        for col in out.columns:
            assert out.schema[col] == nw.Float64
            assert out.schema[col].is_numeric()

    def test_unseen_level_forces_float64_numeric(self) -> None:
        step = StepOneHot(columns=["cat"]).fit(_pd(("cat", ["a", "b"])), _schema())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = nw.from_native(step.transform(_pd(("cat", ["a", "b", "UNSEEN"]))))
        assert out.schema["cat_b"] == nw.Float64
        assert out.schema["cat_b"].is_numeric()


# ---------------------------------------------------------------------------
# Categorical / Enum input dtypes (both in DISCRETE_DTYPES)
# ---------------------------------------------------------------------------


class TestStepOneHotCategoricalEnum:
    def test_categorical_fit_transform(self) -> None:
        df = nw.from_native(
            pl.DataFrame({"x": pl.Series(["a", "b", "c", "a"], dtype=pl.Categorical)}),
            eager_only=True,
        )
        step = StepOneHot(columns=["x"]).fit(df, _schema())
        assert step.schema_out().columns == ["x_a", "x_b", "x_c"]
        out = nw.from_native(step.transform(df))
        assert set(out.columns) == {"x_a", "x_b", "x_c"}

    def test_enum_fit_transform(self) -> None:
        enum = pl.Enum(["a", "b", "c"])
        df = nw.from_native(
            pl.DataFrame({"x": pl.Series(["a", "b", "c", "a"], dtype=enum)}),
            eager_only=True,
        )
        step = StepOneHot(columns=["x"]).fit(df, _schema())
        assert step.schema_out().columns == ["x_a", "x_b", "x_c"]
        out = nw.from_native(step.transform(df))
        assert set(out.columns) == {"x_a", "x_b", "x_c"}


# ---------------------------------------------------------------------------
# Role propagation (ONE_TO_MANY): generated columns inherit the source role
# ---------------------------------------------------------------------------


class TestStepOneHotRoles:
    def test_generated_columns_inherit_source_role(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"]).fit(df, _schema())
        roles = Roles({"x": {"feature"}})
        step.update_roles(roles)
        for col in ("x_a", "x_b", "x_c"):
            assert "feature" in roles.of(col)

    def test_source_role_retained_when_keep_original_cols(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"], keep_original_cols=True).fit(df, _schema())
        roles = Roles({"x": {"feature"}})
        step.update_roles(roles)
        assert "feature" in roles.of("x")

    def test_source_role_removed_when_not_keeping_original(self) -> None:
        df = _pd(("x", ["a", "b", "c"]))
        step = StepOneHot(columns=["x"], keep_original_cols=False).fit(df, _schema())
        roles = Roles({"x": {"feature"}})
        step.update_roles(roles)
        assert roles.of("x") == set()
