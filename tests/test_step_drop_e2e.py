"""End-to-end lifecycle tests for StepDrop: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDrop + narwhals frame
boundary) on both pandas and polars to confirm the pipeline is backend-agnostic.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from shoez import Recipe, StepDrop, StepNormalize
from shoez import Selectors as s


def _data() -> dict[str, list[object]]:
    return {
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [5.0, 1.0, 4.0, 1.0],
        "note": ["p", "q", "r", "s"],
    }


class TestEndToEnd:
    def test_pandas_drops_named_column(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(["note"])).fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_polars_drops_named_column(self) -> None:
        df = pl.DataFrame(_data())
        out = Recipe().add(StepDrop(["note"])).fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_single_name_without_a_list(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop("note")).fit(df).transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_selector_selection(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(s.string_features())).fit(df).transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_drops_multiple_columns(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(["a", "note"])).fit(df).transform(df)
        assert list(out.columns) == ["b"]

    def test_dropping_every_column_yields_an_empty_frame(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(s.everything())).fit(df).transform(df)
        assert list(out.columns) == []

    def test_empty_selection_is_the_identity(self) -> None:
        # Nothing matches ``s.datetime()``, so the step is a no-op.
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(s.datetime())).fit(df).transform(df)
        assert list(out.columns) == ["a", "b", "note"]

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe().add(StepDrop(["note"])).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_unselected_columns_pass_through_unchanged(self) -> None:
        df = pd.DataFrame(_data())
        out = Recipe().add(StepDrop(["note"])).fit(df).transform(df)
        assert out["a"].tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe().add(StepDrop(["note"])).fit(df)
        assert all(step.trained for step in fitted.steps)

    def test_later_step_does_not_see_dropped_columns(self) -> None:
        # The drop happens first, so ``a`` is gone before normalization
        # resolves its own selection.
        df = pd.DataFrame(_data())
        fitted = (
            Recipe()
            .add(StepDrop(["a"]))
            .add(StepNormalize(s.numeric_features()))
            .fit(df)
        )
        assert fitted.steps[1].schema_in().columns == ["b"]
        assert list(fitted.transform(df).columns) == ["b", "note"]

    def test_two_drops_get_distinct_ids(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe().add(StepDrop(["a"])).add(StepDrop(["note"])).fit(df)
        assert [step.id for step in fitted.steps] == ["drop", "drop_2"]
        assert list(fitted.transform(df).columns) == ["b"]
