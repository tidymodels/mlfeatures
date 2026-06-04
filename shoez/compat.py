"""scikit-learn interoperability.

Per ``design/ecosystem.md`` §"Sklearn adapter", a prepped recipe should drop into
:class:`sklearn.pipeline.Pipeline` as a single transformer. :class:`RecipeTransformer`
is that bridge: it wraps the ``fit`` / ``transform`` lifecycle of :class:`~shoez.recipe.Recipe`
without re-implementing it.

The adapter resolves one impedance mismatch between the two worlds. shoez fits on a *single*
frame in which the outcome is identified by its ``"outcome"`` role; sklearn calls
``fit(X, y=None)`` with features and target kept apart and expects ``transform`` to return the
feature matrix only (``y`` flows around the transformer inside a pipeline). So this adapter
re-joins ``y`` onto ``X`` under the recipe's outcome-role column name before fitting, and strips
the outcome column back out of the transformed output.
"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from shoez import _frame
from shoez.recipe import FittedRecipe, Recipe
from shoez.types import DataFrame

__all__ = ["RecipeTransformer"]


class RecipeTransformer(BaseEstimator, TransformerMixin):
    """Wrap a :class:`~shoez.recipe.Recipe` as a scikit-learn transformer.

    Usable as a single step in a :class:`sklearn.pipeline.Pipeline`::

        from shoez.compat import RecipeTransformer

        pipe = Pipeline([
            ("recipe", RecipeTransformer(rec)),
            ("model", LogisticRegression()),
        ])
        pipe.fit(X_train, y_train)

    The recipe is stored verbatim so that :func:`sklearn.base.clone` (and therefore
    cross-validation / grid search) round-trips correctly; :class:`Recipe` is immutable, so
    this is safe.
    """

    def __init__(self, recipe: Recipe):
        # Stored unchanged under the same name as the constructor arg: required by
        # sklearn's get_params / clone contract.
        self.recipe = recipe

    def fit(self, X: DataFrame, y: Any = None) -> "RecipeTransformer":
        """Fit the wrapped recipe on ``X`` (with ``y`` re-joined as the outcome)."""
        data = self._assemble(X, y)
        self.fitted_: FittedRecipe = self.recipe.fit(data)
        self.feature_names_in_ = list(_frame.columns(_frame.from_native(X)))
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Apply the fitted recipe to ``X`` and return the feature matrix only."""
        if not hasattr(self, "fitted_"):
            raise NotFittedError(
                "This RecipeTransformer is not fitted yet; call `fit` before `transform`."
            )
        out = self.fitted_.transform(X)
        return self._drop_outcomes(out)

    # -- internal helpers ---------------------------------------------------

    def _outcome_columns(self) -> list[str]:
        """Outcome-role column names declared on the recipe (may be empty)."""
        return self.recipe.roles.columns_with_role("outcome")

    def _assemble(self, X: DataFrame, y: Any) -> DataFrame:
        """Re-join ``y`` onto ``X`` under the recipe's outcome-role column name.

        With no outcome role (an unsupervised recipe) or no ``y`` (transform time), ``X`` is
        passed through unchanged. A declared outcome role with no ``y`` at fit time, or more
        than one outcome column, is rejected with a clear error.
        """
        outcomes = self._outcome_columns()
        if y is None:
            return X
        if not outcomes:
            raise ValueError(
                "y was provided but the recipe declares no `outcome` role; "
                "assign one with `set_roles(outcome=...)` or pass y=None."
            )
        if len(outcomes) > 1:
            raise ValueError(
                f"multi-outcome recipes are not yet supported (got {outcomes!r}); "
                "declare a single outcome column."
            )
        frame = _frame.from_native(X)
        return _frame.to_native(_frame.with_column(frame, outcomes[0], y))

    def _drop_outcomes(self, data: DataFrame) -> DataFrame:
        """Strip outcome-role columns from transformed output (features only)."""
        outcomes = self._outcome_columns()
        if not outcomes:
            return data
        frame = _frame.from_native(data)
        return _frame.to_native(_frame.apply_drop(frame, outcomes))
