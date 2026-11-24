"""scikit-learn interoperability.

Why: so feature engineering is fitted, cross-validated, and tuned together with
the model, without leaking information from held-out data. A recipe should drop into
`sklearn.pipeline.Pipeline` as a single transformer. `RecipeTransformer`
is that bridge: it wraps an untrained `Recipe` and drives its ``fit`` /
``transform`` lifecycle from sklearn's API without re-implementing it.

The adapter resolves one impedance mismatch between the two worlds. mlfeatures fits on a *single*
data frame in which the target is identified by its ``"target"`` role; sklearn calls
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
from mlfeatures.step import (
    RowOp,
    _step_type_label,  # pyright: ignore[reportPrivateUsage]
)
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

    Use it to put a recipe in front of any scikit-learn model, so the feature
    engineering is trained, cross-validated, and tuned together with the model
    instead of being applied by hand beforehand.

    Usable as a single step in a `sklearn.pipeline.Pipeline`:

    ```python
    from mlfeatures import RecipeTransformer

    pipe = Pipeline([
        ("recipe", RecipeTransformer(recipe)),
        ("model", LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)
    ```

    The wrapped recipe is untrained: ``fit()`` trains it, and ``transform()``
    applies the trained result. The recipe is stored verbatim so that
    `sklearn.base.clone()` (and therefore cross-validation / grid search)
    round-trips correctly; `Recipe` is immutable, so this is safe.

    Step hyperparameters are tunable through sklearn's nested parameter names,
    keyed on step ``id``: ``recipe__<step id>__<parameter>``, for example
    ``GridSearchCV(pipe, {"recipe__recipe__corr__threshold": [0.8, 0.9]})``
    when the transformer is named ``"recipe"`` in the pipeline. A step's ``id``
    and ``columns`` are not tunable.

    It implements `get_feature_names_out()` and therefore composes inside a
    `sklearn.compose.ColumnTransformer` and supports
    ``set_output(transform="pandas" | "polars")``.

    **Target handling.** sklearn keeps ``X`` and ``y`` apart, while a recipe
    identifies the target by its ``"target"`` role. At fit time ``y`` is joined
    onto ``X`` under the recipe's target column name, and ``transform()``
    returns the feature columns only, with the target column removed. Passing
    ``y`` to a recipe with no ``"target"`` role raises a ``ValueError``, as does
    a recipe with more than 1 target column. A step that applies to the
    target column as one of its input columns (for example
    ``StepNormalize(["target"])``) also raises a ``ValueError`` at fit time,
    because sklearn's ``transform()`` never receives ``y``; transform the
    target outside the pipeline instead (for example with
    ``sklearn.compose.TransformedTargetRegressor``). A step that uses the
    target only at fit time, such as ``StepPls`` or ``StepClassDistance``, is
    fine. With no ``y`` (an unsupervised recipe), ``X`` is used as given.

    Parameters
    ----------
    recipe
        The untrained recipe to wrap. Required.

    Attributes
    ----------
    recipe
        The untrained recipe passed to the constructor.
    fitted_
        The trained recipe; set by ``fit()``.
    feature_names_in_
        Column names of the ``X`` seen by ``fit()``.
    n_features_in_
        Number of columns of the ``X`` seen by ``fit()``.
    feature_names_out_
        Column names of the output of ``transform()``.

    Raises
    ------
    NotFittedError
        If ``transform()`` or ``get_feature_names_out()`` is called before
        ``fit()``.

    Examples
    --------
    Wrap a recipe and drive it with sklearn's ``fit`` / ``transform`` API:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepNormalize, StepDummy
    from mlfeatures import Selectors as s
    from mlfeatures import RecipeTransformer
    ```

    ```{python}
    X =pd.DataFrame(
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

    #: The untrained recipe, stored under the constructor argument's name as
    #: sklearn's ``get_params`` / ``clone`` contract requires.
    recipe: Recipe
    #: The trained recipe; set by ``fit()``.
    fitted_: FittedRecipe
    #: Column names seen by ``fit()``, following sklearn's ``feature_names_in_``.
    feature_names_in_: list[str]
    #: Number of columns seen by ``fit()``, following sklearn's ``n_features_in_``.
    n_features_in_: int
    #: Column names the fitted recipe produces; cached by ``fit()``.
    feature_names_out_: list[str]

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
        (use the HTML/diagram view there; see ``_sk_visual_block_``).
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
        """Fit the wrapped recipe on ``X`` (with ``y`` re-joined as the target).

        Inside a ``Pipeline``, ``cross_val_score``, or ``GridSearchCV``, this
        runs on the training portion of each split only, so the statistics the
        recipe learns never see the held-out samples. Applying a recipe by hand
        before splitting would fit them on all samples, which leaks.
        """
        data = self._assemble(X, y)
        fitted = self.recipe.fit(data)
        # Everything is built locally and assigned together at the end, so a
        # fit that raises (here or in the recipe) leaves no half-updated state:
        # ``fitted_`` and the cached names always belong to the same fit.
        if (
            y is not None
            and fitted.schema is not None
            and not fitted.schema.roles.columns_with_role("target")
        ):
            # Only reachable when a role operation using a selector took the
            # target role away, which could not be known before fitting.
            raise ValueError(
                "y was provided but the recipe declares no 'target' role; "
                "a role operation removed it from the column 'y' was attached to"
            )
        if y is not None and fitted.schema is not None:
            # sklearn's transform(X) never carries the column y was attached to,
            # so a later role operation that makes it required at transform
            # (a feature, an id) could never be applied; say so here instead of
            # failing inside the output-name probe below.
            leading = fitted.schema.roles.columns_with_role("target")
            required = [c for c in leading if fitted.schema.transform_required.get(c)]
            if required:
                raise ValueError(
                    f"column(s) {required!r} hold y at fit but a role operation "
                    "written after the first step makes them required at transform; "
                    "sklearn's transform() never receives y, so keep the 'target' "
                    "role or mark the column with "
                    "update_role_requirements(..., transform_required=False)"
                )
        self._check_no_target_steps(fitted)
        names_in = list(_frame.columns(_frame.from_native(X)))
        # Output column names are fixed at fit time (the transform is deterministic),
        # so cache them for get_feature_names_out / set_output. Compute via the fitted
        # recipe directly rather than self.transform, whose set_output wrapper would
        # call get_feature_names_out before the cache exists.
        out = self._drop_targets(
            fitted.transform(X), fitted.roles.columns_with_role("target")
        )
        # A column the recipe does not require at transform (case weights, say)
        # may be missing from the data later, so it cannot be part of an output
        # that is promised to match these names whatever ``X`` carries.
        # A step that reads or rewrites such a column makes it real output, so
        # this is decided from the fitted recipe, never from whether the values
        # happened to come back unchanged on this one training set.
        optional = _optional_columns(fitted)
        names_out = [
            name
            for name in _frame.columns(_frame.from_native(out))
            if name not in optional
        ]
        self.fitted_ = fitted
        self.feature_names_in_ = names_in
        self.n_features_in_ = len(names_in)
        self.feature_names_out_ = names_out
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Apply the fitted recipe to ``X`` and return the feature matrix only.

        It reuses the statistics learned in ``fit()`` and never re-estimates
        them, so a held-out fold is transformed with the training fold's values.
        The output has exactly the columns of ``get_feature_names_out()``, in
        that order, whatever the order of ``X``'s columns or any extra columns
        it carries. A column the recipe does not require at transform time (a
        case-weights column, or one marked with
        ``update_role_requirements(..., transform_required=False)``) is never
        part of the output, so the result does not depend on whether ``X``
        carries it.
        """
        if not hasattr(self, "fitted_"):
            raise NotFittedError(
                "This RecipeTransformer is not fitted yet; call 'fit' before 'transform'."
            )
        out = self.fitted_.transform(X)
        out = self._drop_targets(out, self.fitted_.roles.columns_with_role("target"))
        return self._conform(out)

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> "np.ndarray[Any, Any]":
        """Names of the columns produced by `transform()`.

        ``input_features`` is accepted for sklearn-API compatibility but ignored: the
        output names are determined by the recipe, not the caller. Enables
        `ColumnTransformer` composition and ``set_output``.

        Raises ``NotFittedError`` before ``fit()``.
        """
        if not hasattr(self, "feature_names_out_"):
            raise NotFittedError(
                "This RecipeTransformer is not fitted yet; call 'fit' first."
            )
        return np.asarray(self.feature_names_out_, dtype=object)

    # -- internal helpers ---------------------------------------------------

    def _assemble(self, X: DataFrame, y: Any) -> DataFrame:
        """Re-join ``y`` onto ``X`` under the recipe's target-role column name.

        With no ``y`` (an unsupervised use, or transform time) ``X`` is passed
        through unchanged, even when the recipe declares a target. A ``y`` given
        to a recipe with no target role, more than one target column, a ``y``
        that is not 1-dimensional, or a ``y`` whose length differs from ``X``'s
        is rejected with a clear error.
        """
        # Always the declared view: a stale ``fitted_`` from an earlier fit
        # must not decide how this fit's ``y`` is joined.
        if y is None:
            return X
        targets = self.recipe._declared_targets()  # pyright: ignore[reportPrivateUsage]
        if not targets:
            recipe = self.recipe
            # Only a selector that could hand out the target role makes the
            # target unknowable without data; any other selector operation
            # leaves the plain answer (no target role) intact.
            if recipe._prototype_frame is None and any(  # pyright: ignore[reportPrivateUsage]
                op.position == 0
                and op.plain_names() is None
                and op.role == "target"
                and op.kind != "remove"
                for op in recipe._role_ops  # pyright: ignore[reportPrivateUsage]
            ):
                raise ValueError(
                    "y was provided but the recipe's target cannot be determined: "
                    "a role operation uses a selector, which needs Recipe(data=...) "
                    "to resolve; pass column names or give the recipe prototype data"
                )
            late = [
                op
                for op in recipe._role_ops  # pyright: ignore[reportPrivateUsage]
                if op.role == "target" and op.kind != "remove" and op.position > 0
            ]
            hint = (
                "; a role operation written after the first step cannot define "
                "the target, so declare it with set_roles(target=...) or write "
                "it before the first step"
                if late
                else ""
            )
            raise ValueError(
                "y was provided but the recipe declares no 'target' role; "
                f"assign one with set_roles(target=...) or pass y=None{hint}"
            )
        if len(targets) > 1:
            raise ValueError(
                f"multi-target recipes are not yet supported (got {targets!r}); "
                "declare a single target column"
            )
        frame = _frame.from_native(X)
        values = _as_1d_target(y)
        n_rows, n_values = frame.shape[0], len(values)
        if n_rows != n_values:
            raise ValueError(
                f"X and y have inconsistent numbers of samples: {n_rows} and {n_values}"
            )
        return _frame.to_native(_frame.with_column(frame, targets[0], values))

    def _check_no_target_steps(self, fitted: FittedRecipe) -> None:
        """Reject fitted steps that consume a target column.

        sklearn's ``transform(X)`` never receives ``y``, so a step reading the target
        would fail on the missing column the first time the transformer is applied.
        ``RowOp`` steps and identity (empty-selection) steps never run at transform
        time, so they are allowed. Only columns that were targets before the first
        step count: a column that a later role operation turns into the target was
        an ordinary feature when the step ran. The schema records exactly those,
        with every role operation written before the first step (selectors
        included) already resolved against the training data.
        """
        if fitted.schema is None:
            return
        targets = set(fitted.schema.roles.columns_with_role("target"))
        if not targets:
            return
        for step in fitted.steps:
            if isinstance(step, RowOp) or step._empty_selection:  # pyright: ignore[reportPrivateUsage]
                continue
            used = [c for c in step.schema_in().columns if c in targets]
            if used:
                raise ValueError(
                    f"step {step.id!r} ({_step_type_label(step)}) operates on target "
                    f"column(s) {used!r}; RecipeTransformer cannot apply it because "
                    "sklearn's transform() never receives y; transform the target "
                    "outside the pipeline (e.g. with sklearn's TransformedTargetRegressor)"
                )

    def _drop_targets(self, data: DataFrame, targets: list[str]) -> DataFrame:
        """Strip target-role columns from transformed output (features only).

        ``targets`` comes from the trained recipe's roles, so it includes a
        column a later role operation turned into the target.
        """
        if not targets:
            return data
        frame = _frame.from_native(data)
        return _frame.to_native(_frame.apply_drop(frame, targets))

    def _conform(self, data: DataFrame) -> DataFrame:
        """Select and order the columns as ``feature_names_out_`` says.

        A recipe guarantees which columns exist, not where they sit, and it
        passes unknown extra columns through; sklearn expects a stable layout
        matching ``get_feature_names_out()`` (a model reads columns by
        position, and ``set_output`` relabels them by name list). Columns the
        recipe does not require at transform are not in ``feature_names_out_``,
        so they are dropped here whether or not the data has them.
        """
        frame = _frame.from_native(data)
        wanted = self.feature_names_out_
        present = list(_frame.columns(frame))
        if wanted == present:
            return data
        missing = [name for name in wanted if name not in present]
        if missing:
            raise ValueError(
                f"transform output is missing column(s) {missing!r} that fit() produced"
            )
        return _frame.to_native(frame.select(wanted))


def _optional_columns(fitted: FittedRecipe) -> set[str]:
    """Input columns that may be absent at transform and that no step touches.

    The recipe's schema says which columns ``transform()`` does not require. A
    step that reads or rewrites one of them still makes it real output, so those
    are excluded: ``schema_in()`` of a step with a column selector lists what it
    consumed, and a step without one contributes the columns it adds. Row
    operations are skipped at transform and steps with an empty selection never
    ran, so neither touches anything.
    """
    if fitted.schema is None:
        return set()
    touched: set[str] = set()
    for step in fitted.steps:
        if isinstance(step, RowOp) or step._empty_selection:  # pyright: ignore[reportPrivateUsage]
            continue
        touched.update(step.schema_in().columns)
        touched.update(step.produced_columns())
    return {
        name
        for name, required in fitted.schema.transform_required.items()
        if not required and name not in touched
    }


def _as_1d_target(y: Any) -> Any:
    """``y`` as 1-dimensional values; a single-column 2-D ``y`` is flattened.

    sklearn commonly passes a column vector (``shape (n, 1)``) or a one-column
    data frame; anything wider is a multi-output target, which a recipe's single
    target column cannot hold.
    """
    shape = getattr(y, "shape", None)
    if shape is None:
        return y
    if len(shape) == 0:
        raise ValueError("y must be 1-dimensional or a single column; got a scalar")
    if len(shape) < 2:
        # A Series keeps its dtype; ``with_column`` attaches it to X by position.
        return y
    if len(shape) == 2 and shape[1] == 1:
        # A one-column data frame yields its column as a Series so the dtype
        # (categorical, string, nullable integer) survives.
        if hasattr(y, "iloc"):
            return y.iloc[:, 0]
        if hasattr(y, "to_series"):
            return y.to_series(0)
        if hasattr(y, "to_numpy"):
            return y.to_numpy().ravel()
        return np.asarray(y).ravel()
    raise ValueError(
        f"y must be 1-dimensional or a single column; got shape {tuple(shape)!r}"
    )
