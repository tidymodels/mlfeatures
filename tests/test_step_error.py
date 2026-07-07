"""Tests for ``StepError`` wrapping and chaining (issue #134).

Failures raised from a step's ``fit()`` or ``transform()`` are wrapped in a
``StepError`` that carries the offending step's id, type, and columns, and
chains the original exception so its traceback is preserved.
"""

from __future__ import annotations

import pandas as pd
import pytest

from shoez import Recipe, StepError, StepNormalize


def _data() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})


class TestStepErrorMessage:
    def test_message_includes_id_type_columns(self) -> None:
        err = StepError("norm_1", "step_normalize", ["a", "b"], "fit")
        assert str(err) == (
            "step 'norm_1' (step_normalize) failed during fit; column(s) ['a', 'b']"
        )

    def test_message_omits_columns_when_empty(self) -> None:
        err = StepError("norm_1", "step_normalize", [], "transform")
        assert str(err) == "step 'norm_1' (step_normalize) failed during transform"

    def test_attributes_are_exposed(self) -> None:
        err = StepError("norm_1", "step_normalize", ["a"], "fit")
        assert err.step_id == "norm_1"
        assert err.step_type == "step_normalize"
        assert err.columns == ["a"]
        assert err.phase == "fit"


class TestFitWrapping:
    def test_fit_failure_wrapped_and_chained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(self: StepNormalize, training_data: object, info: object) -> object:
            raise ValueError("boom in fit")

        monkeypatch.setattr(StepNormalize, "fit", boom)
        recipe = Recipe(steps=[StepNormalize(["a", "b"])])
        with pytest.raises(StepError) as excinfo:
            recipe.fit(_data())
        err = excinfo.value
        assert err.step_type == "step_normalize"
        assert err.columns == ["a", "b"]
        assert err.phase == "fit"
        assert isinstance(err.__cause__, ValueError)
        assert str(err.__cause__) == "boom in fit"


class TestTransformWrapping:
    def test_transform_failure_wrapped_and_chained(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recipe = Recipe(steps=[StepNormalize(["a", "b"])])
        fitted = recipe.fit(_data())

        def boom(self: StepNormalize, new_data: object) -> object:
            raise ValueError("boom in transform")

        monkeypatch.setattr(StepNormalize, "transform", boom)
        with pytest.raises(StepError) as excinfo:
            fitted.transform(_data())
        err = excinfo.value
        assert err.step_type == "step_normalize"
        assert err.columns == ["a", "b"]
        assert err.phase == "transform"
        assert isinstance(err.__cause__, ValueError)
        assert str(err.__cause__) == "boom in transform"


def test_existing_step_error_not_double_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inner = StepError("norm_1", "step_normalize", ["a"], "fit")

    def raise_step_error(
        self: StepNormalize, training_data: object, info: object
    ) -> object:
        raise inner

    monkeypatch.setattr(StepNormalize, "fit", raise_step_error)
    recipe = Recipe(steps=[StepNormalize(["a", "b"])])
    with pytest.raises(StepError) as excinfo:
        recipe.fit(_data())
    assert excinfo.value is inner
