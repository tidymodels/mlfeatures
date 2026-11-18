"""Tests for the scikit-learn adapter (``mlfeatures.compat.RecipeTransformer``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Recipe, StepDropCorr, StepFilter, StepLog, StepNormalize
from mlfeatures import Selectors as s
from mlfeatures.compat import RecipeTransformer


def _supervised_recipe() -> Recipe:
    """A recipe that normalizes the features and knows which column is the target."""
    data = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0], "y": [0, 1, 0]})
    return Recipe(data=data, target="y", features=["x1", "x2"]).add(
        StepNormalize(["x1", "x2"])
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

    def test_n_features_in_recorded(self) -> None:
        X, y = _X_y()
        rt = RecipeTransformer(_supervised_recipe()).fit(X, y)
        assert rt.n_features_in_ == 2


class TestAssemble:
    def test_unsupervised_recipe_passes_through(self) -> None:
        data = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        rec = Recipe(data=data).add(StepNormalize(["x1"]))
        X = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        out = RecipeTransformer(rec).fit_transform(X)
        assert list(out.columns) == ["x1"]

    def test_y_without_target_role_raises(self, snapshot: SnapshotAssertion) -> None:
        data = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})
        rec = Recipe(data=data).add(StepNormalize(["x1"]))
        X, y = _X_y()
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(rec).fit(pd.DataFrame({"x1": [1.0, 2.0, 3.0]}), y)
        assert str(excinfo.value) == snapshot


class TestTargetSteps:
    def test_step_on_target_raises(self, snapshot: SnapshotAssertion) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        y = pd.Series([1.0, 10.0, 100.0, 1000.0])
        rec = Recipe(target="price").add(StepLog(s.targets()))
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(rec).fit(X, y)
        assert str(excinfo.value) == snapshot

    def test_row_op_on_target_allowed(self) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 10.0, 100.0, 1000.0])
        rec = Recipe(target="price").add(StepFilter(lambda df: df["price"] > 0))
        RecipeTransformer(rec).fit(X, y)


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


class TestRepr:
    """Printing the transformer and rendering it inside a pipeline diagram."""

    def test_repr_shows_untrained_recipe(self) -> None:
        text = repr(RecipeTransformer(_supervised_recipe()))
        assert text.startswith("RecipeTransformer")
        assert "Recipe (untrained)" in text
        assert "[untrained]" in text

    def test_repr_shows_trained_recipe_after_fit(self) -> None:
        text = repr(RecipeTransformer(_supervised_recipe()).fit(*_X_y()))
        assert "Recipe (trained)" in text
        assert "[trained]" in text
        assert "[untrained]" not in text

    def test_html_repr_labels_transformer_and_embeds_recipe(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.utils import estimator_html_repr

        rt = RecipeTransformer(_supervised_recipe()).fit(*_X_y())
        html = estimator_html_repr(Pipeline([("recipe", rt)]))
        # The diagram labels the block by training state and tucks the recipe
        # into the (expandable) detail rather than inlining it raw. sklearn may
        # truncate the middle of long details, so assert on the clean label and
        # that recipe content is present rather than an exact recipe string.
        assert "RecipeTransformer (trained)" in html
        assert "step_normalize" in html

    def test_visual_block_class_resolves(self) -> None:
        from mlfeatures.compat import _visual_block_class

        assert _visual_block_class() is not None


def _cancer_recipe() -> tuple[Recipe, pd.DataFrame, pd.Series]:
    """A normalize recipe on the breast-cancer dataset, plus its (X, y)."""
    from sklearn.datasets import load_breast_cancer

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    feature_cols = list(X.columns)
    full = X.assign(target=y)
    rec = Recipe(data=full, target="target", features=feature_cols).add(
        StepNormalize(feature_cols)
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


def _tunable_recipe() -> Recipe:
    """Two same-type steps (ids ``corr`` and ``corr_2``) plus a normalize."""
    return (
        Recipe()
        .add(StepDropCorr(["a", "b"], threshold=0.9))
        .add(StepDropCorr(["c", "d"], threshold=0.5))
        .add(StepNormalize(["a"]))
    )


class TestNestedParams:
    def test_get_params_exposes_step_hyperparameters(self) -> None:
        params = RecipeTransformer(_tunable_recipe()).get_params(deep=True)
        assert params["recipe__corr__threshold"] == 0.9
        assert params["recipe__corr__method"] == "pearson"
        assert params["recipe__corr_2__threshold"] == 0.5
        assert "recipe__corr__columns" not in params
        assert "recipe__corr__id" not in params

    def test_shallow_get_params_is_only_the_recipe(self) -> None:
        rec = _tunable_recipe()
        assert RecipeTransformer(rec).get_params(deep=False) == {"recipe": rec}

    def test_set_params_rebuilds_the_recipe(self) -> None:
        rec = _tunable_recipe()
        t = RecipeTransformer(rec)
        assert t.set_params(recipe__corr_2__threshold=0.7) is t
        assert t.get_params()["recipe__corr_2__threshold"] == 0.7
        assert t.get_params()["recipe__corr__threshold"] == 0.9
        assert [s.id for s in t.recipe.steps] == ["corr", "corr_2", "normalize"]
        # the original recipe is untouched
        assert rec.steps[1].get_params()["threshold"] == 0.5  # type: ignore[attr-defined]

    def test_set_params_keeps_roles(self) -> None:
        t = RecipeTransformer(
            Recipe(target="y").add(StepDropCorr(["a", "b"]))
        ).set_params(recipe__corr__threshold=0.3)
        assert t.recipe.roles.columns_with_role("target") == ["y"]

    def test_set_params_whole_recipe_then_nested(self) -> None:
        t = RecipeTransformer(Recipe())
        t.set_params(recipe=_tunable_recipe(), recipe__corr__threshold=0.2)
        assert t.get_params()["recipe__corr__threshold"] == 0.2

    def test_set_params_still_validates(self) -> None:
        t = RecipeTransformer(_tunable_recipe())
        with pytest.raises(ValueError, match="'threshold'"):
            t.set_params(recipe__corr__threshold=2)

    def test_unknown_step_id(self, snapshot: SnapshotAssertion) -> None:
        t = RecipeTransformer(_tunable_recipe())
        with pytest.raises(ValueError) as excinfo:
            t.set_params(recipe__nope__threshold=0.1)
        assert str(excinfo.value) == snapshot

    def test_unknown_parameter(self, snapshot: SnapshotAssertion) -> None:
        t = RecipeTransformer(_tunable_recipe())
        with pytest.raises(ValueError) as excinfo:
            t.set_params(recipe__corr__nope=0.1)
        assert str(excinfo.value) == snapshot

    def test_columns_are_not_tunable(self) -> None:
        t = RecipeTransformer(_tunable_recipe())
        with pytest.raises(ValueError, match="invalid parameter"):
            t.set_params(recipe__corr__columns=["c"])

    def test_malformed_key(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            _tunable_recipe().set_params(threshold=0.1)
        assert str(excinfo.value) == snapshot

    def test_clone_keeps_nested_params(self) -> None:
        from sklearn.base import clone

        t = RecipeTransformer(_tunable_recipe()).set_params(recipe__corr__threshold=0.4)
        cloned = clone(t)
        assert cloned.get_params()["recipe__corr__threshold"] == 0.4
        assert cloned.recipe is not t.recipe

    def test_clone_of_fitted_transformer_is_unfitted(self) -> None:
        from sklearn.base import clone

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.5]})
        fitted = RecipeTransformer(Recipe().add(StepDropCorr(["a", "b"]))).fit(X)
        assert not hasattr(clone(fitted), "fitted_")

    def test_grid_search_over_a_step_hyperparameter(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        rec, X, y = _cancer_recipe()
        feature_cols = list(X.columns)
        rec = rec.add(StepDropCorr(feature_cols, threshold=0.9))
        pipe = Pipeline(
            [
                ("prep", RecipeTransformer(rec)),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        )
        grid = {"prep__recipe__corr__threshold": [0.8, 0.95]}
        search = GridSearchCV(pipe, grid, cv=3)
        search.fit(X, y)
        reported = search.cv_results_["param_prep__recipe__corr__threshold"]
        assert sorted(reported.tolist()) == [0.8, 0.95]
        assert search.best_params_["prep__recipe__corr__threshold"] in (0.8, 0.95)
