"""Recipe-level empty-selection handling (issue #81).

An empty column selection is the identity transform, enforced structurally at
the recipe boundary rather than by per-step guards. These tests pin that
behaviour at the level it now lives: the recipe, not the individual steps. The
per-step ``tests/test_step_*_infrastructure.py`` empty-selection cases remain the
safety net for each step going through the recipe.
"""

from __future__ import annotations

import narwhals as nw
import pandas as pd

from shoez import Recipe, SchemaInfo, StepIntercept, StepNormalize
from shoez.types import Frame


class _RaisingNormalize(StepNormalize):
    """A normalize step that explodes if ``fit`` is ever called.

    Used to prove the recipe never fits a step whose selection is empty.
    """

    def fit(self, training_data: Frame, info: SchemaInfo) -> "_RaisingNormalize":
        raise AssertionError("fit() must not be called on an empty selection")


def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})


class TestEmptySelectionIdentity:
    def test_empty_selection_is_passthrough(self) -> None:
        df = _df()
        out = Recipe(steps=[StepNormalize([])]).fit(df).transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_fit_not_called_on_empty_selection(self) -> None:
        df = _df()
        # _RaisingNormalize.fit() raises; reaching it would fail the test.
        out = Recipe(steps=[_RaisingNormalize([])]).fit(df).transform(df)
        pd.testing.assert_frame_equal(out, df)

    def test_empty_selected_step_kept_and_marked(self) -> None:
        df = _df()
        fitted = Recipe(steps=[StepNormalize([])]).fit(df)
        # The step stays in the fitted pipeline (inspectable), marked identity.
        assert len(fitted.steps) == 1
        step = fitted.steps[0]
        assert step._empty_selection is True  # pyright: ignore[reportPrivateUsage]
        assert step.trained is True

    def test_non_empty_selection_still_runs(self) -> None:
        df = _df()
        out = Recipe(steps=[StepNormalize(["a"])]).fit(df).transform(df)
        # ``a`` is normalized (mean ~0); ``b`` passes through untouched.
        assert abs(out["a"].mean()) < 1e-9
        assert out["b"].tolist() == [10.0, 20.0, 30.0]


class TestIntercept:
    def test_intercept_runs_when_present(self) -> None:
        df = _df()
        # StepIntercept has no input selector and must never be skipped.
        out = Recipe(steps=[StepIntercept()]).fit(df).transform(df)
        assert "intercept" in out.columns
        assert out["intercept"].tolist() == [1.0, 1.0, 1.0]

    def test_intercept_runs_alongside_empty_step(self) -> None:
        df = _df()
        recipe = Recipe(steps=[StepNormalize([]), StepIntercept()])
        out = recipe.fit(df).transform(df)
        # The empty normalize is a no-op; the intercept still runs.
        assert "intercept" in out.columns
        assert list(out.columns) == ["a", "b", "intercept"]


class TestRefitResetsMarker:
    def test_marker_resets_when_selection_later_matches(self) -> None:
        # A dtype selector that matches nothing in an all-string frame but
        # matches in a numeric frame — exercises the per-fit marker reset.
        recipe = Recipe(steps=[StepNormalize(nw.selectors.numeric())])

        strings = pd.DataFrame({"s": ["x", "y", "z"]})
        first = recipe.fit(strings)
        assert first.steps[0]._empty_selection is True  # pyright: ignore[reportPrivateUsage]

        numbers = _df()
        second = recipe.fit(numbers)
        # Same Recipe object, new data: the step is no longer empty and runs.
        assert second.steps[0]._empty_selection is False  # pyright: ignore[reportPrivateUsage]
        out = second.transform(numbers)
        assert abs(out["a"].mean()) < 1e-9
