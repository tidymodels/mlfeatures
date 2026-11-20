"""Edge-case tests for ``RecipeTransformer``: state, ``y`` handling, layout, backends."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from syrupy.assertion import SnapshotAssertion

from mlfeatures import Recipe, StepDummy, StepFilter, StepImputeMean, StepNormalize
from mlfeatures import Selectors as s
from mlfeatures.compat import RecipeTransformer

DF = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x2": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        "c": list("aabbab"),
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)
X = DF.drop(columns="y")
Y = DF["y"]

# set_output(transform="polars") arrived in scikit-learn 1.4; the declared floor is 1.3
needs_polars_output = pytest.mark.skipif(
    tuple(int(p) for p in sklearn.__version__.split(".")[:2]) < (1, 4),
    reason="set_output(transform='polars') needs scikit-learn >= 1.4",
)


def _recipe() -> Recipe:
    return Recipe(target="y").add(StepNormalize(["x1", "x2"]))


def _full_recipe() -> Recipe:
    return _recipe().add(StepDummy(["c"]))


class TestFitIsAtomic:
    """A fit that raises must not leave half-updated state behind."""

    def _bad_recipe(self) -> Recipe:
        # A step on the target column: rejected after the recipe has fitted.
        return Recipe(target="y").add(StepNormalize(["y"]))

    def test_failed_first_fit_leaves_transformer_unfitted(self) -> None:
        t = RecipeTransformer(self._bad_recipe())
        with pytest.raises(ValueError):
            t.fit(X, Y)
        assert not hasattr(t, "fitted_")
        with pytest.raises(NotFittedError):
            t.transform(X)
        with pytest.raises(NotFittedError):
            t.get_feature_names_out()

    def test_failed_refit_keeps_the_previous_fit_consistent(self) -> None:
        t = RecipeTransformer(_full_recipe()).fit(X, Y)
        names = list(t.get_feature_names_out())
        fitted = t.fitted_
        t.recipe = self._bad_recipe()
        with pytest.raises(ValueError):
            t.fit(X, Y)
        assert t.fitted_ is fitted
        assert list(t.get_feature_names_out()) == names
        assert list(t.transform(X).columns) == names

    def test_failed_refit_in_the_recipe_keeps_previous_fit(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        fitted = t.fitted_
        with pytest.raises(Exception):
            t.fit(X.drop(columns="x1"), Y)
        assert t.fitted_ is fitted
        assert t.n_features_in_ == 3

    def test_refit_replaces_all_state(self) -> None:
        t = RecipeTransformer(Recipe(target="y"))
        assert t.fit(X, Y) is t
        t.fit(X.drop(columns="c"), Y)
        assert t.feature_names_in_ == ["x1", "x2"]
        assert t.n_features_in_ == 2
        assert list(t.get_feature_names_out()) == ["x1", "x2"]

    def test_fit_does_not_mutate_the_recipe(self) -> None:
        recipe = _recipe()
        before = repr(recipe)
        RecipeTransformer(recipe).fit(X, Y)
        assert repr(recipe) == before

    def test_two_transformers_from_one_recipe_are_independent(self) -> None:
        recipe = _recipe()
        a = RecipeTransformer(recipe).fit(X, Y)
        b = RecipeTransformer(recipe).fit(X.assign(x1=X["x1"] * 10), Y)
        assert a.transform(X)["x1"].iloc[0] != b.transform(X)["x1"].iloc[0]


class TestOutputLayout:
    """The output columns always match ``get_feature_names_out()``."""

    def test_column_order_follows_the_fit(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        out = t.transform(X[["c", "x2", "x1"]])
        assert list(out.columns) == list(t.get_feature_names_out())

    def test_values_stay_with_their_column_when_order_differs(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        shuffled = t.transform(X[["c", "x2", "x1"]])
        pd.testing.assert_frame_equal(shuffled, t.transform(X))

    def test_extra_columns_are_dropped(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        out = t.transform(X.assign(z=1))
        assert list(out.columns) == list(t.get_feature_names_out())

    def test_target_in_transform_input_is_dropped(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        assert list(t.transform(DF).columns) == ["x1", "x2", "c"]

    def test_set_output_pandas_does_not_mislabel_reordered_input(self) -> None:
        t = RecipeTransformer(_recipe()).set_output(transform="pandas").fit(X, Y)
        out = t.transform(X[["c", "x2", "x1"]])
        assert out["c"].tolist() == X["c"].tolist()
        assert out["x1"].iloc[0] < 0

    def test_set_output_pandas_with_extra_column(self) -> None:
        t = RecipeTransformer(_recipe()).set_output(transform="pandas").fit(X, Y)
        out = t.transform(X.assign(z=1))
        assert list(out.columns) == ["x1", "x2", "c"]

    @needs_polars_output
    def test_set_output_polars_does_not_mislabel_reordered_input(self) -> None:
        t = RecipeTransformer(_recipe()).set_output(transform="polars").fit(X, Y)
        out = t.transform(X[["c", "x2", "x1"]])
        assert out["c"].to_list() == X["c"].tolist()

    def test_missing_input_column_raises(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        with pytest.raises(ValueError):
            t.transform(X.drop(columns="x1"))

    def test_pandas_index_is_preserved(self) -> None:
        shuffled = X.sample(frac=1, random_state=0)
        t = RecipeTransformer(_recipe()).fit(shuffled, Y.loc[shuffled.index])
        assert t.transform(shuffled).index.tolist() == shuffled.index.tolist()

    def test_zero_and_one_row_transform(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        assert t.transform(X.iloc[:0]).shape == (0, 3)
        assert t.transform(X.iloc[:1]).shape == (1, 3)

    def test_row_ops_do_not_change_the_row_count(self) -> None:
        recipe = Recipe(target="y").add(StepFilter(lambda df: df["x1"] > 3.0))
        t = RecipeTransformer(recipe).fit(X, Y)
        assert t.transform(X).shape[0] == len(X)
        assert t.fit_transform(X, Y).shape[0] == len(X)


class TestTargetValues:
    """How ``y`` is accepted and joined."""

    @pytest.mark.parametrize(
        "y",
        [
            Y,
            Y.to_numpy(),
            list(Y),
            pl.Series(Y),
            Y.to_numpy().reshape(-1, 1),
            Y.to_frame(),
        ],
        ids=["series", "ndarray", "list", "polars-series", "column-vector", "frame"],
    )
    def test_accepted_forms_of_y(self, y: Any) -> None:
        t = RecipeTransformer(_recipe()).fit(X, y)
        assert t.transform(X).shape == (6, 3)

    def test_y_is_joined_by_position_not_by_index(self) -> None:
        shuffled = X.sample(frac=1, random_state=0)
        y_reset = Y.loc[shuffled.index].reset_index(drop=True)
        t = RecipeTransformer(_recipe())
        joined = t._assemble(shuffled, y_reset)
        assert joined["y"].tolist() == y_reset.tolist()
        assert t.fit(shuffled, y_reset).transform(shuffled).shape == (6, 3)

    def test_polars_x_with_pandas_and_numpy_y(self) -> None:
        Xp = pl.from_pandas(X)
        for y in (Y, Y.to_numpy()):
            out = RecipeTransformer(_recipe()).fit(Xp, y).transform(Xp)
            assert isinstance(out, pl.DataFrame)
            assert out.shape == (6, 3)

    def test_cross_backend_y_keeps_nulls_and_dtype(self) -> None:
        Xp = pl.from_pandas(X)
        nullable = pd.Series([1, 2, None, 4, 5, 6], dtype="Int64")
        joined = RecipeTransformer(_recipe())._assemble(Xp, nullable)
        assert joined["y"].dtype == pl.Int64
        assert joined["y"].to_list() == [1, 2, None, 4, 5, 6]

        cat = pd.Series(["u", "v", None, "u", "v", "u"], dtype="category")
        joined = RecipeTransformer(_recipe())._assemble(Xp, cat)
        assert joined["y"].dtype == pl.Categorical
        assert joined["y"].to_list() == ["u", "v", None, "u", "v", "u"]

    def test_wide_y_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(_recipe()).fit(X, np.c_[Y, Y])
        assert str(excinfo.value) == snapshot

    def test_scalar_y_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(_recipe()).fit(X, np.array(5))
        assert str(excinfo.value) == snapshot

    def test_length_mismatch_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(_recipe()).fit(X, Y.iloc[:3])
        assert str(excinfo.value) == snapshot

    def test_categorical_and_string_targets(self) -> None:
        y = pd.Series(list("ababab"))
        for target in (y, y.astype("category")):
            out = RecipeTransformer(_recipe()).fit(X, target).transform(X)
            assert out.shape == (6, 3)

    def test_missing_values_in_y_are_accepted(self) -> None:
        y = Y.copy()
        y.iloc[0] = np.nan
        assert RecipeTransformer(_recipe()).fit(X, y).transform(X).shape == (6, 3)

    def test_declared_target_without_y_passes_x_through(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X)
        assert list(t.transform(X).columns) == ["x1", "x2", "c"]

    def test_y_overwrites_a_target_column_already_in_x(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X.assign(y=-999.0), Y)
        # the target is stripped from the output either way
        assert "y" not in t.transform(X.assign(y=-999.0)).columns

    def test_numpy_input_is_rejected(self) -> None:
        with pytest.raises(TypeError):
            RecipeTransformer(_recipe()).fit(X.to_numpy(), Y)
        t = RecipeTransformer(_recipe()).fit(X, Y)
        with pytest.raises(TypeError):
            t.transform(X.to_numpy())

    def test_multi_target_raises(self, snapshot: SnapshotAssertion) -> None:
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(Recipe(target=["y", "x1"])).fit(X, Y)
        assert str(excinfo.value) == snapshot


class TestTargetDeclaredByRoleOps:
    def test_leading_plain_name_add_role(self) -> None:
        recipe = Recipe().add_role("y", role="target").add(StepNormalize(["x1"]))
        assert RecipeTransformer(recipe).fit(X, Y).transform(X).shape == (6, 3)

    def test_leading_update_role_with_prototype(self) -> None:
        recipe = Recipe(data=DF).update_role("y", new_role="target")
        out = RecipeTransformer(recipe.add(StepNormalize(["x1"]))).fit(X, Y)
        assert out.transform(X).shape == (6, 3)

    def test_leading_remove_role_removes_the_target(
        self, snapshot: SnapshotAssertion
    ) -> None:
        recipe = Recipe(target="y").remove_role("y", role="target")
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(recipe).fit(X, Y)
        assert str(excinfo.value) == snapshot

    def test_late_target_op_gets_a_specific_hint(
        self, snapshot: SnapshotAssertion
    ) -> None:
        recipe = Recipe().add(StepNormalize(["x1"])).add_role("y", role="target")
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(recipe).fit(X, Y)
        assert str(excinfo.value) == snapshot

    def test_selector_that_could_define_the_target_needs_data(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # Without prototype data the selector cannot be resolved, and it might
        # be what names the target, so the target cannot be told.
        recipe = Recipe().add_role(s.starts_with("y"), role="target")
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(recipe).fit(X, Y)
        assert str(excinfo.value) == snapshot

    def test_selector_that_cannot_define_the_target_is_not_blamed(
        self, snapshot: SnapshotAssertion
    ) -> None:
        # A selector handing out another role, or taking one away, cannot make
        # a target, so the plain answer (no target role) is the true one.
        with pytest.raises(ValueError) as added:
            RecipeTransformer(Recipe().add_role(s.numeric_features(), role="g")).fit(
                X, Y
            )
        with pytest.raises(ValueError) as removed:
            RecipeTransformer(
                Recipe().remove_role(s.numeric_features(), role="feature")
            ).fit(X, Y)
        assert str(added.value) == snapshot
        assert str(removed.value) == str(added.value)


class TestSelectorTargetsAreChecked:
    """A step reading a target is rejected whichever way the target was named."""

    def test_selector_target_read_by_a_step_is_rejected(
        self, snapshot: SnapshotAssertion
    ) -> None:
        recipe = (
            Recipe()
            .add_role(s.starts_with("y"), role="target")
            .add(StepNormalize(["y"]))
        )
        with pytest.raises(ValueError) as excinfo:
            RecipeTransformer(recipe).fit(DF)
        assert str(excinfo.value) == snapshot

    def test_message_matches_the_plain_declaration(self) -> None:
        by_selector = (
            Recipe()
            .add_role(s.starts_with("y"), role="target")
            .add(StepNormalize(["y"]))
        )
        declared = Recipe(target="y").add(StepNormalize(["y"]))
        with pytest.raises(ValueError) as expected:
            RecipeTransformer(declared).fit(DF)
        with pytest.raises(ValueError) as actual:
            RecipeTransformer(by_selector).fit(DF)
        assert str(actual.value) == str(expected.value)

    def test_target_made_by_a_later_operation_does_not_count(self) -> None:
        recipe = (
            Recipe()
            .add(StepNormalize(["y"]))
            .update_role(s.starts_with("y"), new_role="target", old_role="feature")
        )
        RecipeTransformer(recipe).fit(DF)


class TestOptionalColumns:
    """Columns the recipe does not require at transform are never output."""

    XW = X.assign(w=1.0)

    def _recipe(self) -> Recipe:
        return Recipe(target="y", case_weights="w").add(StepNormalize(["x1", "x2"]))

    def test_names_leave_out_the_case_weights(self) -> None:
        t = RecipeTransformer(self._recipe()).fit(self.XW, Y)
        assert list(t.get_feature_names_out()) == ["x1", "x2", "c"]

    def test_output_matches_the_names_with_or_without_the_column(self) -> None:
        t = RecipeTransformer(self._recipe()).fit(self.XW, Y)
        names = list(t.get_feature_names_out())
        assert list(t.transform(self.XW).columns) == names
        assert list(t.transform(X).columns) == names

    def test_nan_case_weights_are_still_optional(self) -> None:
        xw = X.assign(w=[np.nan] + [1.0] * (len(X) - 1))
        t = RecipeTransformer(self._recipe()).fit(xw, Y)
        names = list(t.get_feature_names_out())
        assert names == ["x1", "x2", "c"]
        assert list(t.transform(X).columns) == names

    def test_set_output_pandas_without_the_column(self) -> None:
        t = RecipeTransformer(self._recipe()).set_output(transform="pandas")
        t.fit(self.XW, Y)
        assert list(t.transform(X).columns) == ["x1", "x2", "c"]

    @needs_polars_output
    def test_set_output_polars_without_the_column(self) -> None:
        t = RecipeTransformer(self._recipe()).set_output(transform="polars")
        t.fit(self.XW, Y)
        assert t.transform(X).columns == ["x1", "x2", "c"]

    def test_column_made_optional_by_an_override(self) -> None:
        # Features are declared so that "c" keeps only its custom role.
        recipe = (
            Recipe(target="y", features=["x1", "x2"])
            .set_roles(id="c")
            .update_role_requirements("id", transform_required=False)
            .add(StepNormalize(["x1"]))
        )
        t = RecipeTransformer(recipe).fit(X, Y)
        assert list(t.get_feature_names_out()) == ["x1", "x2"]
        assert list(t.transform(X.drop(columns="c")).columns) == ["x1", "x2"]

    def test_required_columns_are_still_required(self) -> None:
        t = RecipeTransformer(self._recipe()).fit(self.XW, Y)
        with pytest.raises(ValueError):
            t.transform(X.drop(columns="x1"))

    def test_a_step_that_leaves_the_weights_unchanged_still_makes_them_output(
        self,
    ) -> None:
        # The imputer finds nothing to fill in the training data, so "w" comes
        # back identical; the decision must come from the recipe, not from that.
        recipe = Recipe(target="y", case_weights="w").add(StepImputeMean(["x1", "w"]))
        t = RecipeTransformer(recipe).fit(self.XW, Y)
        assert list(t.get_feature_names_out()) == ["x1", "x2", "c", "w"]
        new = self.XW.assign(w=[np.nan] + [1.0] * (len(X) - 1))
        out = t.transform(new)
        assert list(out.columns) == ["x1", "x2", "c", "w"]
        assert out["w"].notna().all()

    def test_a_step_that_reads_the_weights_needs_them_at_transform(self) -> None:
        recipe = Recipe(target="y", case_weights="w").add(StepImputeMean(["x1", "w"]))
        t = RecipeTransformer(recipe).fit(self.XW, Y)
        with pytest.raises(Exception):  # noqa: B017 - the recipe's own error
            t.transform(X)

    def test_a_step_that_adds_columns_leaves_the_weights_optional(self) -> None:
        recipe = Recipe(target="y", case_weights="w").add(StepDummy(["c"]))
        t = RecipeTransformer(recipe).fit(self.XW, Y)
        names = list(t.get_feature_names_out())
        assert "w" not in names
        assert list(t.transform(X).columns) == names
        assert list(t.transform(self.XW).columns) == names

    def test_a_row_operation_reading_the_weights_leaves_them_optional(self) -> None:
        # The filter reads "w" at fit time only; it is skipped at transform.
        recipe = (
            Recipe(target="y", case_weights="w")
            .add(StepFilter(lambda df: df["w"] > 0.0))
            .add(StepNormalize(["x1", "x2"]))
        )
        t = RecipeTransformer(recipe).fit(self.XW, Y)
        assert list(t.get_feature_names_out()) == ["x1", "x2", "c"]
        assert list(t.transform(X).columns) == ["x1", "x2", "c"]

    def test_a_step_with_an_empty_selection_leaves_the_weights_optional(self) -> None:
        recipe = (
            Recipe(target="y", case_weights="w")
            .add(StepNormalize(["x1", "x2"]))
            .add(StepImputeMean(s.has_role("nonexistent")))
        )
        t = RecipeTransformer(recipe).fit(self.XW, Y)
        assert list(t.get_feature_names_out()) == ["x1", "x2", "c"]
        assert list(t.transform(X).columns) == ["x1", "x2", "c"]

    def test_weights_with_missing_values_in_other_forms_stay_optional(self) -> None:
        # None in an object column, NaN in a float column: neither changes
        # whether the column is optional, which no longer depends on values.
        for weights in ([None, 1.0, 1.0, 1.0, 1.0, 1.0], [np.nan] * 6):
            xw = X.assign(w=weights)
            t = RecipeTransformer(self._recipe()).fit(xw, Y)
            assert list(t.get_feature_names_out()) == ["x1", "x2", "c"]
            assert list(t.transform(X).columns) == ["x1", "x2", "c"]

    def test_polars_input_with_optional_weights(self) -> None:
        xw = pl.from_pandas(self.XW)
        t = RecipeTransformer(self._recipe()).fit(xw, Y)
        assert t.transform(pl.from_pandas(X)).columns == ["x1", "x2", "c"]


class TestConformReportsMissingColumns:
    def test_missing_column_raises_a_clear_error(
        self, snapshot: SnapshotAssertion
    ) -> None:
        t = RecipeTransformer(_full_recipe()).fit(X, Y)
        broken = t.transform(X).drop(columns="x1")
        with pytest.raises(ValueError) as excinfo:
            t._conform(broken)  # pyright: ignore[reportPrivateUsage]
        assert str(excinfo.value) == snapshot

    def test_every_missing_column_is_named(self) -> None:
        t = RecipeTransformer(_full_recipe()).fit(X, Y)
        broken = t.transform(X).drop(columns=["x1", "x2"])
        with pytest.raises(ValueError) as excinfo:
            t._conform(broken)  # pyright: ignore[reportPrivateUsage]
        assert "'x1'" in str(excinfo.value)
        assert "'x2'" in str(excinfo.value)

    def test_a_complete_output_is_only_reordered(self) -> None:
        t = RecipeTransformer(_full_recipe()).fit(X, Y)
        out = t.transform(X)
        shuffled = out[list(reversed(out.columns))]
        conformed = t._conform(shuffled)  # pyright: ignore[reportPrivateUsage]
        assert list(conformed.columns) == list(out.columns)


class TestSklearnIntegration:
    def test_unsupervised_recipe_rejects_y(self) -> None:
        # y reaches every transformer in a Pipeline / ColumnTransformer, so a
        # recipe without a target role cannot sit in front of a supervised model
        recipe = Recipe().add(StepNormalize(["x1", "x2"]))
        with pytest.raises(ValueError):
            ColumnTransformer([("r", RecipeTransformer(recipe), ["x1", "x2"])]).fit(
                X, Y
            )

    def test_column_transformer_passes_y_through(self) -> None:
        ct = ColumnTransformer([("r", RecipeTransformer(_recipe()), ["x1", "x2"])])
        assert ct.fit(X, Y).transform(X).shape == (6, 2)

    def test_cross_val_score_with_shuffled_index_and_numpy_y(self) -> None:
        shuffled = X.sample(frac=1, random_state=1)
        pipe = Pipeline(
            [("r", RecipeTransformer(_full_recipe())), ("m", LinearRegression())]
        )
        scores = cross_val_score(
            pipe, shuffled, Y.loc[shuffled.index].to_numpy(), cv=KFold(3)
        )
        assert np.allclose(scores, 1.0)

    def test_cross_val_score_with_polars(self) -> None:
        pipe = Pipeline(
            [("r", RecipeTransformer(_full_recipe())), ("m", LinearRegression())]
        )
        scores = cross_val_score(pipe, pl.from_pandas(X), Y.to_numpy(), cv=KFold(3))
        assert np.allclose(scores, 1.0)

    def test_classification_with_string_target(self) -> None:
        y = pd.Series(list("ababab"))
        pipe = Pipeline(
            [("r", RecipeTransformer(_full_recipe())), ("m", LogisticRegression())]
        ).fit(X, y)
        assert pipe.predict(X).shape == (6,)
        assert pipe.predict_proba(X).shape == (6, 2)

    def test_pipeline_prefix_transform(self) -> None:
        pipe = Pipeline(
            [("r", RecipeTransformer(_recipe())), ("m", LinearRegression())]
        ).fit(X.drop(columns="c"), Y)
        assert pipe[:-1].transform(X.drop(columns="c")).shape == (6, 2)

    def test_grid_search_with_parallel_jobs(self) -> None:
        pipe = Pipeline(
            [("r", RecipeTransformer(_full_recipe())), ("m", LinearRegression())]
        )
        grid = GridSearchCV(
            pipe, {"m__fit_intercept": [True, False]}, cv=KFold(3), n_jobs=2
        ).fit(X, Y)
        assert "m__fit_intercept" in grid.best_params_


class TestEstimatorState:
    def test_check_is_fitted(self) -> None:
        t = RecipeTransformer(_recipe())
        with pytest.raises(NotFittedError):
            check_is_fitted(t)
        check_is_fitted(t.fit(X, Y))

    def test_clone_of_fitted_is_unfitted(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        with pytest.raises(NotFittedError):
            clone(t).transform(X)

    def test_pickle_round_trip(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        again = pickle.loads(pickle.dumps(t))
        pd.testing.assert_frame_equal(again.transform(X), t.transform(X))

    def test_deepcopy_is_independent(self) -> None:
        t = RecipeTransformer(_recipe()).fit(X, Y)
        other = copy.deepcopy(t)
        assert other.fitted_ is not t.fitted_
        pd.testing.assert_frame_equal(other.transform(X), t.transform(X))

    def test_non_string_column_names_do_not_corrupt_state(self) -> None:
        X3 = X.copy()
        X3.columns = pd.Index([0, 1, 2])
        t = RecipeTransformer(Recipe()).fit(X3)
        assert t.n_features_in_ == 3
        assert t.transform(X3).shape == (6, 3)

    def test_get_params_is_just_the_recipe(self) -> None:
        assert sorted(RecipeTransformer(_recipe()).get_params(deep=False)) == ["recipe"]

    def test_set_params_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError):
            RecipeTransformer(_recipe()).set_params(bogus=1)


class TestLateRoleOpMakesTargetRequired:
    """A late role operation must not turn the column y was attached to into a required input."""

    @pytest.mark.parametrize(
        "late",
        [
            lambda r: r.update_role("y", new_role="feature"),
            lambda r: r.update_role("y", new_role="id"),
            lambda r: r.remove_role("y", role="target"),
        ],
    )
    def test_fit_explains_the_problem(self, late: Any) -> None:
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        y = pd.Series([1.0, 2.0, 3.0, 5.0], name="y")
        recipe = late(Recipe(target="y").add(StepNormalize(["a"])))
        with pytest.raises(ValueError, match="required at transform"):
            RecipeTransformer(recipe).fit(X, y)
