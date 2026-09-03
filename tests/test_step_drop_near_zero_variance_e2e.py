"""End-to-end lifecycle tests for StepDropNearZeroVariance: Recipe -> fit -> transform.

Exercises the full vertical slice (Recipe driver + StepDropNearZeroVariance +
narwhals frame boundary) on both pandas and polars to confirm the pipeline is
backend-agnostic.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import Mode, Recipe, SchemaInfo, StepDropNearZeroVariance


def _data() -> dict[str, list[float]]:
    # ``sparse`` is near-zero-variance: 1 value repeated 999 times, 1 outlier.
    # ``a`` and ``b`` vary enough to be kept.
    return {
        "a": [float(i) for i in range(1000)],
        "sparse": [1.0] * 999 + [0.0],
        "b": [float(i % 7) for i in range(1000)],
    }


class TestEndToEnd:
    def test_pandas_drops_sparse_column(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropNearZeroVariance(["a", "sparse", "b"])])
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_polars_drops_sparse_column(self) -> None:
        df = pl.DataFrame(_data())
        recipe = Recipe().add(StepDropNearZeroVariance(["a", "sparse", "b"]))
        out = recipe.fit(df).transform(df)

        assert isinstance(out, pl.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_no_drop_when_everything_varies(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropNearZeroVariance(["a", "b"])])
        out = recipe.fit(df).transform(df)
        # a/b both vary enough, so nothing is dropped; sparse passes through.
        assert list(out.columns) == ["a", "sparse", "b"]

    def test_string_sparse_column_dropped(self) -> None:
        # Near-zero variance is type-agnostic — a sparse string column drops.
        df = pd.DataFrame(
            {"s": ["x"] * 999 + ["y"], "t": [str(i % 7) for i in range(1000)]}
        )
        recipe = Recipe(steps=[StepDropNearZeroVariance(["s", "t"])])
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["t"]

    def test_unique_cut_argument_changes_outcome(self) -> None:
        df = pd.DataFrame(_data())
        # sparse has pct_unique = 100 * 2 / 1000 = 0.2; a unique_cut below
        # that keeps it, since the unique-value share no longer clears it.
        recipe = Recipe(steps=[StepDropNearZeroVariance(["sparse"], unique_cut=0.1)])
        out = recipe.fit(df).transform(df)
        assert "sparse" in out.columns

    def test_freq_cut_argument_changes_outcome(self) -> None:
        df = pd.DataFrame(_data())
        # sparse has freq_ratio = 999; a freq_cut above that keeps it.
        recipe = Recipe(steps=[StepDropNearZeroVariance(["sparse"], freq_cut=1000)])
        out = recipe.fit(df).transform(df)
        assert "sparse" in out.columns

    def test_transform_is_idempotent(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropNearZeroVariance(["a", "sparse", "b"])]).fit(df)
        first = fitted.transform(df)
        second = fitted.transform(df)
        pd.testing.assert_frame_equal(first, second)

    def test_extra_columns_outside_selection_are_preserved(self) -> None:
        df = pd.DataFrame(_data())
        recipe = Recipe(steps=[StepDropNearZeroVariance(["a", "sparse"])])
        out = recipe.fit(df).transform(df)
        assert list(out.columns) == ["a", "b"]

    def test_trained_flag_set_by_recipe(self) -> None:
        df = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropNearZeroVariance(["a", "sparse"])]).fit(df)
        assert all(step.trained for step in fitted.steps)

    def test_transform_uses_train_decision_not_new_data(self) -> None:
        train = pd.DataFrame(_data())
        fitted = Recipe(steps=[StepDropNearZeroVariance(["a", "sparse", "b"])]).fit(
            train
        )
        kept = set(fitted.transform(train).columns)
        # New data where ``sparse`` now varies a lot must not resurrect it —
        # the drop decision is fixed at fit time.
        new = pd.DataFrame(
            {
                "a": [1.0, 2.0],
                "sparse": [7.0, 9.0],
                "b": [5.0, 1.0],
            }
        )
        assert set(fitted.transform(new).columns) == kept


class TestStepDropNearZeroVarianceGuards:
    def test_transform_before_fit_raises(self) -> None:
        step = StepDropNearZeroVariance(["a", "sparse"])
        with pytest.raises(RuntimeError):
            step.transform(nw.from_native(pd.DataFrame(_data())))

    def test_accepts_any_dtype(self) -> None:
        accepted = StepDropNearZeroVariance(["a", "sparse"]).accepts_dtypes()
        assert nw.Float64 in accepted
        assert nw.String in accepted

    def test_fit_returns_self(self) -> None:
        step = StepDropNearZeroVariance(["a", "sparse"])
        fitted = step.fit(nw.from_native(pd.DataFrame(_data())), SchemaInfo())
        assert fitted is step

    def test_default_id(self) -> None:
        assert StepDropNearZeroVariance(["a"]).id == "nzv"

    def test_explicit_id(self) -> None:
        assert StepDropNearZeroVariance(["a"], id="my_nzv").id == "my_nzv"

    def test_mode_is_removing(self) -> None:
        assert StepDropNearZeroVariance(["a", "sparse"]).mode is Mode.REMOVING

    def test_not_sparse(self) -> None:
        assert StepDropNearZeroVariance(["a"]).produces_sparse is False

    def test_unique_cut_too_high_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropNearZeroVariance(["a"], unique_cut=101)
        assert str(excinfo.value) == snapshot

    def test_unique_cut_too_low_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropNearZeroVariance(["a"], unique_cut=-1)
        assert str(excinfo.value) == snapshot

    def test_negative_freq_cut_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            StepDropNearZeroVariance(["a"], freq_cut=-1)
        assert str(excinfo.value) == snapshot

    def test_default_freq_cut_and_unique_cut(self) -> None:
        step = StepDropNearZeroVariance(["a"])
        assert step.freq_cut == pytest.approx(95 / 5)
        assert step.unique_cut == 10
