"""Tests for the scikit-learn adapter (``shoez.compat.RecipeTransformer``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez import Recipe, StepNormalize
from shoez.compat import RecipeTransformer


def _supervised_recipe() -> Recipe:
    """A recipe that normalizes the predictors and knows which column is the outcome."""
    data = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0, 1, 0]})
    return (
        Recipe(data=data)
        .set_roles(outcome="y", predictor=["x1", "x2"])
        .add(StepNormalize(["x1", "x2"]))
    )


def _X_y() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
    y = np.array([0, 1, 0])
    return X, y


class TestRoundTrip:
    def test_transform_returns_features_only(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).fit(X, y)
        out = rt.transform(X)
        assert "y" not in out.columns
        assert list(out.columns) == ["x1", "x2"]

    def test_fit_transform_matches_fit_then_transform(self) -> None:
        X, y = _X_y()
        a = RecipeTransformer(_supervised_recipe()).fit_transform(X, y)
        b = RecipeTransformer(_supervised_recipe()).fit(X, y).transform(X)
        pd.testing.assert_frame_equal(a, b)

    def test_normalization_applied(self) -> None:
        X, y = _X_y()
        out = RecipeTransformer(_supervised_recipe()).fit_transform(X, y)
        # standardized columns have ~zero mean
        assert out["x1"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_feature_names_in_recorded(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).fit(X, y)
        assert rt.feature_names_in_ == ["x1", "x2"]


class TestAssemble:
    def test_unsupervised_recipe_passes_through(self) -> None:
        data = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        rec = Recipe(data=data).add(StepNormalize(["x1"]))
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        out = RecipeTransformer(rec).fit_transform(X)
        assert list(out.columns) == ["x1"]

    def test_y_without_outcome_role_raises(self) -> None:
        data = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        rec = Recipe(data=data).add(StepNormalize(["x1"]))
        X, y = _X_y()
        with pytest.raises(ValueError, match="no `outcome` role"):
            RecipeTransformer(rec).fit(pd.DataFrame({"x1": [1.0, 2.0, 3.0]}), y)


class TestBackendParity:
    def test_polars_in_polars_out(self) -> None:
        X = pl.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        y = [0, 1, 0]
        out = RecipeTransformer(_supervised_recipe()).fit_transform(X, y)
        assert isinstance(out, pl.DataFrame)
        assert out.columns == ["x1", "x2"]


class TestSklearnContract:
    def test_clone_yields_unfitted_equivalent(self) -> None:
        from sklearn.base import clone

        rt = RecipeTransformer(_supervised_recipe()).fit(*_X_y())
        cloned = clone(rt)
        assert isinstance(cloned, RecipeTransformer)
        assert not hasattr(cloned, "fitted_")
