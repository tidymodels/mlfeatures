"""Recipe-level ``accepts_dtypes()`` enforcement (issue #200).

Each step declares the dtypes it is valid to apply to; the recipe checks the
resolved selection against that declaration before calling ``step.fit()``. These
tests pin the check at the level it lives: the recipe, not the individual steps.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from shoez import (
    ANY_DTYPE,
    AcceptedDtypes,
    Recipe,
    SchemaInfo,
    StepDropZeroVariance,
    StepDummy,
    StepNormalize,
)
from shoez.types import Frame


class _RaisingNormalize(StepNormalize):
    """A normalize step that explodes if ``fit`` is ever called."""

    def fit(self, training_data: Frame, info: SchemaInfo) -> "_RaisingNormalize":
        raise AssertionError("fit() must not be called on a rejected selection")


class _AnyDtypeNormalize(StepNormalize):
    """A normalize step that opts out of dtype validation."""

    def accepts_dtypes(self) -> AcceptedDtypes:
        return ANY_DTYPE


def _mixed() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "label": ["x", "y", "x"]})


class TestDtypeRejection:
    def test_numeric_step_rejects_string_column(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            Recipe(steps=[StepNormalize(["label"])]).fit(_mixed())
        assert str(excinfo.value) == snapshot

    def test_error_lists_every_offending_column(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pl.DataFrame({"a": [1.0, 2.0], "label": ["x", "y"], "flag": [True, False]})
        with pytest.raises(TypeError) as excinfo:
            Recipe(steps=[StepNormalize(["a", "label", "flag"])]).fit(df)
        assert str(excinfo.value) == snapshot

    def test_categorical_step_rejects_float_column(
        self, snapshot: SnapshotAssertion
    ) -> None:
        with pytest.raises(TypeError) as excinfo:
            Recipe(steps=[StepDummy(["a"])]).fit(_mixed())
        assert str(excinfo.value) == snapshot

    def test_check_runs_before_fit(self) -> None:
        with pytest.raises(TypeError):
            Recipe(steps=[_RaisingNormalize(["label"])]).fit(_mixed())

    def test_error_names_a_custom_step_id(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(TypeError) as excinfo:
            Recipe(steps=[StepNormalize(["label"], id="my_step")]).fit(_mixed())
        assert str(excinfo.value) == snapshot


class TestDtypeAcceptance:
    def test_accepted_dtypes_pass(self) -> None:
        df = _mixed()
        out = Recipe(steps=[StepNormalize(["a"])]).fit(df).transform(df)
        assert out["a"].tolist() == pytest.approx([-1.0, 0.0, 1.0])

    def test_any_dtype_skips_validation(self) -> None:
        # ANY_DTYPE opts out, so the step is fitted and the failure (if any)
        # comes from the step itself rather than from the recipe's guard.
        with pytest.raises(Exception) as excinfo:
            Recipe(steps=[_AnyDtypeNormalize(["label"])]).fit(_mixed())
        assert not isinstance(excinfo.value, TypeError)

    def test_any_dtype_step_accepts_strings(self) -> None:
        df = _mixed()
        out = Recipe(steps=[StepDropZeroVariance(["a", "label"])]).fit(df).transform(df)
        assert list(out.columns) == ["a", "label"]

    def test_empty_selection_is_not_checked(self) -> None:
        df = _mixed()
        out = Recipe(steps=[StepNormalize([])]).fit(df).transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_transform_does_not_recheck(self) -> None:
        # The ptype already pins new data's dtypes, so the guard is fit-only.
        df = _mixed()
        fitted = Recipe(steps=[StepNormalize(["a"])]).fit(df)
        assert fitted.transform(df) is not None
