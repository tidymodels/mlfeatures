"""End-to-end lifecycle tests for StepFilter: Recipe -> fit -> transform.

Exercises the ``RowOp`` contract through the full vertical slice on both pandas
and polars: the filter reshapes the *training* frame at fit time (so downstream
steps estimate on the filtered rows) but is skipped at ``transform()`` time so
no rows are dropped from data being scored.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from shoez import Mode, Recipe, StepFilter, StepNormalize


def _data() -> dict[str, list[float]]:
    return {"age": [25.0, 41.0, 33.0, 52.0], "x": [1.0, 2.0, 3.0, 4.0]}


class TestSkipAtTransform:
    def test_pandas_transform_keeps_all_rows(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepFilter(lambda d: d["age"] > 30)).fit(df).transform(df)
        assert isinstance(out, pd.DataFrame)
        assert out.shape[0] == 4  # transform never filters
        assert list(out.columns) == ["age", "x"]

    def test_polars_transform_keeps_all_rows(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepFilter(lambda d: d["age"] > 30)).fit(df).transform(df)
        assert isinstance(out, pl.DataFrame)
        assert out.shape[0] == 4


class TestFilterAffectsFit:
    """A RowOp reshapes the training frame, so a downstream step estimates on the
    filtered rows — the whole point of applying row ops at fit time."""

    def test_downstream_stats_use_filtered_rows(self) -> None:
        df = pd.DataFrame(_data())
        with_filter = (
            Recipe()
            .add(StepFilter(lambda d: d["age"] > 30))
            .add(StepNormalize(["age"]))
            .fit(df)
            .transform(df)
        )
        without_filter = Recipe().add(StepNormalize(["age"])).fit(df).transform(df)
        # The filter changes which rows feed StepNormalize's mean/std, so the
        # normalized column differs from the unfiltered pipeline.
        assert with_filter["age"].tolist() != without_filter["age"].tolist()


class TestStepFilter:
    def test_mode_is_row_op(self) -> None:
        assert StepFilter(lambda d: d["age"] > 0).mode is Mode.ROW_OP

    def test_is_row_op_subclass(self) -> None:
        from shoez import RowOp

        assert isinstance(StepFilter(lambda d: d["age"] > 0), RowOp)

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepFilter(lambda d: d["age"] > 30)]).fit(df)
        assert all(step.trained for step in fitted.steps)
