"""End-to-end lifecycle tests for StepLog: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepLog + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import math

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, Recipe, SchemaInfo, StepLog


def _data() -> dict[str, list[float]]:
    # ``zero`` is out of the logarithm's domain; ``a`` and ``b`` are not.
    return {
        "a": [1.0, 10.0, 100.0, 1000.0],
        "zero": [0.0, 1.0, 2.0, 3.0],
        "b": [1.0, 2.0, 4.0, 8.0],
    }


class TestEndToEnd:
    def test_pandas_takes_the_natural_log(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepLog(["a", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "zero", "b"]  # structure preserved
        assert out["a"].tolist() == pytest.approx([math.log(v) for v in _data()["a"]])

    def test_polars_takes_the_natural_log(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepLog(["a", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert out["b"].to_list() == pytest.approx([math.log(v) for v in _data()["b"]])

    def test_base_ten_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe().add(StepLog(["a"], base=10.0))
        out = recipe.fit(df).transform(df)
        assert out["a"].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])

    def test_offset_through_recipe(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe().add(StepLog(["zero"], base=2.0, offset=1.0))
        out = recipe.fit(df).transform(df)
        assert out["zero"].tolist() == pytest.approx([0.0, 1.0, math.log2(3), 2.0])

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepLog(["a"])])
        out = recipe.fit(df).transform(df)
        assert out["b"].tolist() == _data()["b"]
        assert out["zero"].tolist() == _data()["zero"]

    def test_out_of_domain_column_warns_and_yields_negative_infinity(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepLog(["a", "zero"])])
        with pytest.warns(UserWarning) as record:
            fitted = recipe.fit(df)
        assert str(record[0].message) == snapshot
        with pytest.warns(UserWarning):
            out = fitted.transform(df)
        assert out["zero"].tolist()[0] == float("-inf")

    def test_transform_uses_no_state_from_new_data(self) -> None:
        train = pd.DataFrame({"a": [1.0, 10.0]})
        fitted = Recipe(steps=[StepLog(["a"], base=10.0)]).fit(train)
        out = fitted.transform(pd.DataFrame({"a": [100.0, 1000.0]}))
        # Elementwise: the same values as if they had been in the training data.
        assert out["a"].tolist() == pytest.approx([2.0, 3.0])

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepLog(["a", "b"])]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_integer_in_float_out_through_recipe(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3, 4]})  # integer column
        out = Recipe(steps=[StepLog(["a"])]).fit(df).transform(df)
        assert out["a"].dtype == float  # promoted to float by the logarithm

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepLog(["a", "b"])]).fit(pd.DataFrame(_data()))
        assert all(step.trained for step in fitted.steps)


class TestStepLogUnit:
    def test_mode_is_one_to_one(self) -> None:
        assert StepLog(["a"]).mode is Mode.ONE_TO_ONE

    def test_default_id(self) -> None:
        assert StepLog(["a"]).id == "log"

    def test_explicit_id(self) -> None:
        assert StepLog(["a"], id="my_log").id == "my_log"

    def test_default_base_is_e(self) -> None:
        assert StepLog(["a"]).base == math.e

    def test_fit_returns_self(self) -> None:
        step = StepLog(["a", "b"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_accepts_numeric_dtypes(self) -> None:
        accepted = StepLog(["a"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.Int64 in accepted
        assert nw.String not in accepted

    def test_schema_out_marks_logged_columns_float(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        step = StepLog(["a", "b"])
        step.fit(nw.from_native(df), SchemaInfo())
        out_schema = step.schema_out()
        assert out_schema.dtypes["a"] == "Float64"  # int in -> float out
        assert out_schema.dtypes["b"] == "Float64"

    def test_schema_in_reports_the_input_dtypes(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        step = StepLog(["a"])
        step.fit(nw.from_native(df), SchemaInfo())
        assert step.schema_in().dtypes["a"] == "Int64"

    def test_not_sparse(self) -> None:
        assert StepLog(["a"]).produces_sparse is False
