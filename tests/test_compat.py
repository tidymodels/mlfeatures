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


class TestFeatureNamesOut:
    def test_matches_transform_columns(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).fit(X, y)
        assert list(rt.get_feature_names_out()) == list(rt.transform(X).columns)
        assert list(rt.get_feature_names_out()) == ["x1", "x2"]

    def test_ignores_input_features_argument(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).fit(X, y)
        assert list(rt.get_feature_names_out(["ignored"])) == ["x1", "x2"]

    def test_raises_before_fit(self) -> None:
        from sklearn.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            RecipeTransformer(_supervised_recipe()).get_feature_names_out()


class TestSetOutput:
    def test_set_output_pandas(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).set_output(transform="pandas")
        out = rt.fit(X, y).transform(X)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["x1", "x2"]


class TestColumnTransformer:
    def test_composes_in_column_transformer(self) -> None:
        from sklearn.compose import ColumnTransformer

        X, y = _X_y()
        ct = ColumnTransformer(
            [("rec", RecipeTransformer(_supervised_recipe()), ["x1", "x2"])]
        )
        out = ct.fit_transform(X, y)
        assert out.shape == (3, 2)
        assert list(ct.get_feature_names_out()) == ["rec__x1", "rec__x2"]


class TestEstimatorContract:
    def test_get_and_set_params(self) -> None:
        rec = _supervised_recipe()
        rt = RecipeTransformer(rec)
        assert rt.get_params()["recipe"] is rec
        other = _supervised_recipe()
        rt.set_params(recipe=other)
        assert rt.recipe is other

    def test_transform_before_fit_raises(self) -> None:
        from sklearn.exceptions import NotFittedError

        X, _ = _X_y()
        with pytest.raises(NotFittedError):
            RecipeTransformer(_supervised_recipe()).transform(X)

    def test_fitted_attributes_have_trailing_underscore(self) -> None:
        # sklearn convention: learned state is exposed via trailing-underscore attrs.
        rt = RecipeTransformer(_supervised_recipe()).fit(*_X_y())
        assert hasattr(rt, "fitted_")
        assert hasattr(rt, "feature_names_in_")
        assert hasattr(rt, "feature_names_out_")


def _cancer_recipe() -> tuple[Recipe, pd.DataFrame, pd.Series]:
    """A normalize recipe on the breast-cancer dataset, plus its (X, y)."""
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    features = list(X.columns)
    full = X.assign(target=y)
    rec = (
        Recipe(data=full)
        .set_roles(outcome="target", predictor=features)
        .add(StepNormalize(features))
    )
    return rec, X, y


class TestEndToEnd:
    def test_pipeline_with_model(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        rec, X, y = _cancer_recipe()
        pipe = Pipeline(
            [
                ("recipe", RecipeTransformer(rec)),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
        pipe.fit(X, y)
        # normalization gives the model an easy, well-conditioned problem
        assert pipe.score(X, y) > 0.9

    def test_grid_search_cv(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        rec, X, y = _cancer_recipe()
        pipe = Pipeline(
            [
                ("recipe", RecipeTransformer(rec)),
                ("model", LogisticRegression(max_iter=1000)),
            ]
        )
        search = GridSearchCV(pipe, {"model__C": [0.1, 1.0]}, cv=3)
        search.fit(X, y)  # exercises clone() of RecipeTransformer across folds
        assert search.best_score_ > 0.9
        assert search.best_params_["model__C"] in (0.1, 1.0)
