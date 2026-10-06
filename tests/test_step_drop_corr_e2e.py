"""End-to-end lifecycle tests for StepDropCorr: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDropCorr + narwhals
frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals.stable.v2 as nw
import pandas as pd
import polars as pl
import pytest

from mlfeatures import Mode, Recipe, SchemaInfo, StepDropCorr


def _data() -> dict[str, list[float]]:
    # ``b`` is a perfect linear function of ``a`` (corr == 1.0); ``c`` is
    # uncorrelated with both.
    return {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],
        "c": [5.0, 1.0, 4.0, 1.0, 9.0],
    }


class TestEndToEnd:
    def test_pandas_drops_one_of_a_correlated_pair(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        # Exactly one of the correlated pair {a, b} is removed; c is kept.
        assert "c" in out.columns
        assert len({"a", "b"} & set(out.columns)) == 1
        assert len(out.columns) == 2

    def test_polars_drops_one_of_a_correlated_pair(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepDropCorr(["a", "b", "c"], threshold=0.9))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert "c" in out.columns
        assert len({"a", "b"} & set(out.columns)) == 1

    def test_no_drop_when_below_threshold(self) -> None:
        df = pd.DataFrame(_data())
        # A threshold above the max pairwise correlation drops nothing.
        recipe = Recipe(steps=[StepDropCorr(["a", "c"], threshold=0.99)])
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["a", "b", "c"]

    def test_unselected_columns_pass_through(self) -> None:
        df = pd.DataFrame(_data())
        # b is outside the selection, so it is never a drop candidate.
        recipe = Recipe(steps=[StepDropCorr(["a", "c"], threshold=0.9)])
        out = recipe.fit(df).transform(df)
        assert "b" in out.columns

    def test_transform_uses_train_decision_not_new_data(self) -> None:
        train = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)]).fit(train)
        kept = set(fitted.transform(train).columns)
        # New data with different correlation structure must not change which
        # columns are dropped — the decision is fixed at fit time.
        new = pd.DataFrame({"a": [1.0, 9.0], "b": [9.0, 1.0], "c": [3.0, 3.0]})
        assert set(fitted.transform(new).columns) == kept

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropCorr(["a", "b", "c"], threshold=0.9)]).fit(df)
        pd.testing.assert_frame_equal(fitted.transform(df), fitted.transform(df))

    def test_trained_flag_set_by_recipe(self) -> None:
        fitted = Recipe(steps=[StepDropCorr(["a", "b", "c"])]).fit(
            pd.DataFrame(_data())
        )
        assert all(step.trained for step in fitted.steps)


class TestStepDropCorrUnit:
    def test_mode_is_removing(self) -> None:
        assert StepDropCorr(["a", "b"]).mode is Mode.REMOVING

    def test_default_id(self) -> None:
        assert StepDropCorr(["a", "b"]).id == "corr"

    def test_explicit_id(self) -> None:
        assert StepDropCorr(["a", "b"], id="my_corr").id == "my_corr"

    def test_fit_returns_self(self) -> None:
        step = StepDropCorr(["a", "b", "c"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_transform_before_fit_raises(self) -> None:
        step = StepDropCorr(["a", "b"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_not_sparse(self) -> None:
        assert StepDropCorr(["a", "b"]).produces_sparse is False
