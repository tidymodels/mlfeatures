"""scikit-learn interoperability.

Per ``design/ecosystem.md`` §"Sklearn adapter", a trained recipe should drop into
`sklearn.pipeline.Pipeline` as a single transformer. `RecipeTransformer`
is that bridge: it wraps the ``fit`` / ``transform`` lifecycle of `Recipe`
without re-implementing it.

The adapter resolves one impedance mismatch between the two worlds. mlfeatures fits on a *single*
frame in which the target is identified by its ``"target"`` role; sklearn calls
``fit(X, y=None)`` with features and target kept apart and expects ``transform`` to return the
feature matrix only (``y`` flows around the transformer inside a pipeline). So this adapter
re-joins ``y`` onto ``X`` under the recipe's target-role column name before fitting, and strips
the target column back out of the transformed output.
"""

from __future__ import annotations

import importlib
from typing import Any, cast

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from mlfeatures import _frame
from mlfeatures.recipe import FittedRecipe, Recipe
from mlfeatures.types import DataFrame

__all__ = ["RecipeTransformer"]


def _visual_block_class() -> Any:
    """sklearn's private ``_VisualBlock``, located across version layouts.

    The helper moved from ``sklearn.utils._estimator_html_repr`` to
    ``sklearn.utils._repr_html.estimator`` in sklearn 1.9. We resolve it
    dynamically so `RecipeTransformer._sk_visual_block_` keeps working on both;
    returns ``None`` if neither layout is present.
    """
    for module_name in (
        "sklearn.utils._estimator_html_repr",
        "sklearn.utils._repr_html.estimator",
    ):
        try:
            return importlib.import_module(module_name)._VisualBlock
        except (ImportError, AttributeError):  # pragma: no cover - version drift
            continue
    return None  # pragma: no cover - unknown sklearn layout


class RecipeTransformer(BaseEstimator, TransformerMixin):
    """Wrap a `Recipe` as a scikit-learn transformer.

    Usable as a single step in a `sklearn.pipeline.Pipeline`:

    ```python
    from mlfeatures.compat import RecipeTransformer

    pipe = Pipeline([
        ("recipe", RecipeTransformer(rec)),
        ("model", LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)
    ```

    The recipe is stored verbatim so that `sklearn.base.clone()` (and therefore
    cross-validation / grid search) round-trips correctly; `Recipe` is immutable, so
    this is safe.

    Step hyperparameters are tunable through sklearn's nested parameter names,
    keyed on step ``id``: ``recipe__<step id>__<parameter>``, for example
    ``GridSearchCV(pipe, {"prep__recipe__corr__threshold": [0.8, 0.9]})``. A
    step's ``id`` and ``columns`` are not tunable.

    It implements `get_feature_names_out()` and therefore composes inside a
    `sklearn.compose.ColumnTransformer` and supports
    ``set_output(transform="pandas" | "polars")``.

    Examples
    --------
    Wrap a recipe and drive it with sklearn's ``fit`` / ``transform`` API:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepNormalize, StepDummy
    from mlfeatures import Selectors as s
    from mlfeatures.compat import RecipeTransformer
    ```

    ```{python}
    X = pd.DataFrame(
        {"age": [25, 41, 33, 52], "city": ["NYC", "LA", "NYC", "SF"]}
    )
    rec = (
        Recipe()
        .add(StepNormalize(s.numeric_features()))
        .add(StepDummy(s.nominal_features()))
    )
    fitted = RecipeTransformer(rec).fit(X)
    fitted.transform(X)
    ```

    `get_feature_names_out()` reports the produced feature names, so the
    transformer composes inside a `sklearn.compose.ColumnTransformer`:

    ```{python}
    fitted.get_feature_names_out()
    ```
    """

    def __init__(self, recipe: Recipe):
        # Stored unchanged under the same name as the constructor arg: required by
        # sklearn's get_params / clone contract.
        self.recipe = recipe

    def set_params(self, **params: Any) -> "RecipeTransformer":
        """Set parameters, including step hyperparameters as ``recipe__<step id>__<param>``.

        A whole ``recipe`` is applied first, then the nested keys on top of it.
        `Recipe` is immutable, so nested keys rebuild ``self.recipe`` instead of
        mutating it (sklearn's default ``set_params`` would discard the result).
        """
        nested = {
            key.removeprefix("recipe__"): params.pop(key)
            for key in list(params)
            if key.startswith("recipe__")
        }
        cast(Any, super()).set_params(**params)
        if nested:
            self.recipe = self.recipe.set_params(**nested)
        return self

    def _display_recipe(self) -> Recipe | FittedRecipe:
        """The recipe to show when printing: the fitted one once available.

        Before ``fit`` only the untrained ``recipe`` exists; after ``fit`` the
        trained ``fitted_`` is the more informative view (resolved columns,
        ``[trained]`` markers), so prefer it.
        """
        fitted = getattr(self, "fitted_", None)
        return fitted if fitted is not None else self.recipe

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """A clean text view: the wrapped recipe's ``repr``, indented under a header.

        sklearn's default estimator ``repr`` would inline the recipe's multi-line
        ``repr`` as a constructor argument, which reads poorly. We show the recipe
        block as-is instead (the trained view once fitted). Note this is only used
        when printing the transformer directly; inside a ``Pipeline`` the text
        ``repr`` is produced by sklearn's pretty-printer, which we cannot override
        (use the HTML/diagram view there — see ``_sk_visual_block_``).
        """
        indented = "\n".join(
            "  " + line for line in repr(self._display_recipe()).splitlines()
        )
        return f"RecipeTransformer\n{indented}"

    def _sk_visual_block_(self) -> Any:
        """Render as a single, labelled block in sklearn's HTML/diagram repr.

        This is what shows in Jupyter / VS Code / Colab when a `Pipeline`
        containing this transformer is displayed, and via
        ``sklearn.utils.estimator_html_repr``. The block is labelled with the
        training state and expands to the wrapped recipe's full ``repr``, so the
        pipeline diagram stays readable instead of inlining the recipe's
        multi-line text. Falls back to sklearn's default block if the private
        ``_VisualBlock`` helper cannot be located on this sklearn version.
        """
        visual_block = _visual_block_class()
        if visual_block is None:  # pragma: no cover - unknown sklearn layout
            return cast(Any, super())._sk_visual_block_()

        state = "trained" if getattr(self, "fitted_", None) is not None else "untrained"
        return visual_block(
            "single",
            self,
            names=f"RecipeTransformer ({state})",
            name_details=repr(self._display_recipe()),
        )

    def fit(self, X: DataFrame, y: Any = None) -> "RecipeTransformer":
        """Fit the wrapped recipe on ``X`` (with ``y`` re-joined as the target)."""
        data = self._assemble(X, y)
        self.fitted_: FittedRecipe = self.recipe.fit(data)
        self.feature_names_in_ = list(_frame.columns(_frame.from_native(X)))
        # Output column names are fixed at fit time (the transform is deterministic),
        # so cache them for get_feature_names_out / set_output. Compute via the fitted
        # recipe directly rather than self.transform, whose set_output wrapper would
        # call get_feature_names_out before the cache exists.
        out = self._drop_targets(self.fitted_.transform(X))
        self.feature_names_out_ = list(_frame.columns(_frame.from_native(out)))
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Apply the fitted recipe to ``X`` and return the feature matrix only."""
        if not hasattr(self, "fitted_"):
            raise NotFittedError(
                "This RecipeTransformer is not fitted yet; call 'fit' before 'transform'."
            )
        out = self.fitted_.transform(X)
        return self._drop_targets(out)

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> "np.ndarray[Any, Any]":
        """Names of the columns produced by `transform()`.

        ``input_features`` is accepted for sklearn-API compatibility but ignored: the
        output names are determined by the recipe, not the caller. Enables
        `ColumnTransformer` composition and ``set_output``.
        """
        if not hasattr(self, "feature_names_out_"):
            raise NotFittedError(
                "This RecipeTransformer is not fitted yet; call 'fit' first."
            )
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- internal helpers ---------------------------------------------------

    def _target_columns(self) -> list[str]:
        """Target-role column names declared on the recipe (may be empty)."""
        return self.recipe.roles.columns_with_role("target")

    def _assemble(self, X: DataFrame, y: Any) -> DataFrame:
        """Re-join ``y`` onto ``X`` under the recipe's target-role column name.

        With no target role (an unsupervised recipe) or no ``y`` (transform time), ``X`` is
        passed through unchanged. A declared target role with no ``y`` at fit time, or more
        than one target column, is rejected with a clear error.
        """
        targets = self._target_columns()
        if y is None:
            return X
        if not targets:
            raise ValueError(
                "y was provided but the recipe declares no 'target' role; "
                "assign one with set_roles(target=...) or pass y=None"
            )
        if len(targets) > 1:
            raise ValueError(
                f"multi-target recipes are not yet supported (got {targets!r}); "
                "declare a single target column"
            )
        frame = _frame.from_native(X)
        return _frame.to_native(_frame.with_column(frame, targets[0], y))

    def _drop_targets(self, data: DataFrame) -> DataFrame:
        """Strip target-role columns from transformed output (features only)."""
        targets = self._target_columns()
        if not targets:
            return data
        frame = _frame.from_native(data)
        return _frame.to_native(_frame.apply_drop(frame, targets))
