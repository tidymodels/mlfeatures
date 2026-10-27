"""Partial least squares feature extraction: fit + apply.

Like `mlfeatures.step_pca`, a *set* of numeric input columns is replaced by a new
*set* of component columns, so every output descends from the whole input set
rather than a single source column. Unlike PCA, the components are chosen to
be predictive of a target column, the way `mlfeatures.step_class_distance` reads a
target at fit time without treating it as a selected input feature.

The *fit* half (`fit_pls()`) delegates the actual NIPALS computation to
``sklearn.cross_decomposition.PLSRegression`` and copies only the plain NumPy
arrays it needs out of the fitted estimator into a frozen `PlsFit`; the
*apply* half (`apply_pls()`) reuses those arrays to project any frame onto the
learned components, exactly as `mlfeatures.step_pca` does for its loadings: the
basis is never recomputed from new data, giving train/serve parity.
`StepPls` wraps both into the `Step` lifecycle so a `Recipe` can fit and apply
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np

from mlfeatures import _frame
from mlfeatures._dependencies import import_optional
from mlfeatures.step import Step
from mlfeatures.types import (
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["PlsFit", "fit_pls", "apply_pls", "StepPls"]


@dataclass(frozen=True)
class PlsFit:
    """Partial-least-squares basis learned at fit time.

    Bundles everything `apply_pls()` needs to project new data onto the same
    components the training set defined: the input column order, the
    centering mean, and the rotation matrix. Marked ``frozen=True`` so the
    fitted state cannot be mutated and drift train-to-serve.

    Attributes
    ----------
    columns
        Input column names in the order the rotation rows correspond to.
    mean
        Training-set mean of each input column (length ``n_features``).
        Applied as ``X - mean`` before projection.
    rotation
        Matrix of shape ``(n_features, num_comp)`` mapping centered input
        columns to component scores: new scores are ``(X - mean) @
        rotation``. Copied from ``PLSRegression.x_rotations_``.
    num_comp
        Number of components actually produced (``rotation.shape[1]``),
        capped at the number of input columns and the number of complete
        rows.
    """

    columns: list[str]
    mean: np.ndarray[Any, Any]
    rotation: np.ndarray[Any, Any] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    num_comp: int = 0


def fit_pls(selected: Frame, target: "nw.Series[Any]", num_comp: int) -> PlsFit:
    """Learn the PLS centering mean and rotation matrix from ``selected``.

    Rows with any missing feature value or a null target are dropped before
    fitting, so the basis is estimated on complete cases only. The number of
    components is capped at the number of input columns and the number of
    complete rows. The math itself is delegated to
    ``sklearn.cross_decomposition.PLSRegression``; only its plain NumPy
    arrays are kept.

    A numeric ``target`` is fit directly (PLS regression). A categorical or
    string ``target`` is dummy-coded into one indicator column per class,
    sorted by `sorted()`, before fitting (PLS discriminant analysis); the
    dummy-coding is transparent to `apply_pls()`, which only ever projects
    the feature columns.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the numeric columns to project.
        Native frames are wrapped via `mlfeatures._frame.from_native()` at the
        recipe boundary.
    target
        Narwhals ``Series`` of a numeric or categorical target, aligned
        row-for-row with ``selected``.
    num_comp
        Number of components to keep. Must be an ``int`` strictly greater
        than zero. Booleans are rejected even though ``bool`` is an ``int``
        subclass in Python -- ``num_comp=True`` is almost always a bug.

    Returns
    -------
    PlsFit
        The fitted basis. Pair with `apply_pls()` to project a frame.

    Raises
    ------
    ValueError
        If ``num_comp`` is not a positive integer.
    """
    if (
        isinstance(num_comp, bool)
        or not isinstance(num_comp, int)  # pyright: ignore[reportUnnecessaryIsInstance]
        or num_comp <= 0
    ):
        raise ValueError(
            f"'num_comp' must be an integer greater than zero, got {num_comp!r}"
        )
    columns = _frame.columns(selected)
    x = _frame.to_float_array(selected)
    feature_complete = cast("np.ndarray[Any, Any]", ~np.isnan(x).any(axis=1))

    if target.dtype.is_numeric():
        y = _frame.series_to_float_array(target)
        complete = feature_complete & ~np.isnan(y)
        x_complete = x[complete]
        y_complete = y[complete]
    else:
        x_complete, y_complete = _dummy_code_target(x, feature_complete, target)

    eff = min(num_comp, x_complete.shape[1], x_complete.shape[0])

    cross_decomposition = import_optional(
        "sklearn.cross_decomposition", purpose="to fit PLS components"
    )
    pls = cross_decomposition.PLSRegression(n_components=eff, scale=False)
    pls.fit(x_complete, y_complete)

    mean = np.asarray(pls._x_mean, dtype=np.float64)  # pyright: ignore[reportPrivateUsage]
    rotation = np.asarray(pls.x_rotations_, dtype=np.float64)
    return PlsFit(columns=columns, mean=mean, rotation=rotation, num_comp=eff)


def _dummy_code_target(
    x: "np.ndarray[Any, Any]",
    feature_complete: "np.ndarray[Any, Any]",
    target: "nw.Series[Any]",
) -> tuple["np.ndarray[Any, Any]", "np.ndarray[Any, Any]"]:
    """Restrict ``x`` to complete rows and dummy-code a categorical ``target``.

    One indicator column per class (sorted by `sorted()`, matching
    `mlfeatures.step_class_distance.fit_class_distance`) turns PLS discriminant
    analysis into an ordinary multi-output PLS regression -- the same
    ``x_rotations_`` extraction in `fit_pls` works unchanged.
    """
    target_np = target.to_numpy()
    not_null = np.array([v is not None and not _is_nan(v) for v in target_np])
    complete = feature_complete & not_null
    x_complete = x[complete]
    labels = target_np[complete]
    classes: list[Any] = sorted(set(labels.tolist()))
    y_complete = np.zeros((len(labels), len(classes)), dtype=np.float64)
    for j, cls in enumerate(classes):
        y_complete[:, j] = labels == cls
    return x_complete, y_complete


def _is_nan(value: Any) -> bool:
    """``True`` for a floating-point NaN, and for any value unequal to itself."""
    try:
        return bool(value != value)
    except TypeError:  # pragma: no cover
        return False


def apply_pls(
    data: Frame, fit: PlsFit, prefix: str, keep_original_cols: bool = False
) -> Frame:
    """Replace the fitted input columns with their PLS component scores.

    The input columns named in ``fit`` are dropped (unless
    ``keep_original_cols``) and ``fit.num_comp`` score columns named
    ``{prefix}1 ... {prefix}{num_comp}`` are appended, computed as ``(X -
    fit.mean) @ fit.rotation``. Rows with a missing input value yield ``NaN``
    scores. The basis comes from ``fit`` -- it is not recomputed from
    ``data``.

    Parameters
    ----------
    data
        Internal `Frame` to transform. Must contain the columns in ``fit``.
    fit
        Output of `fit_pls()`: the centering mean and rotation matrix.
    prefix
        Prefix for the produced component column names.
    keep_original_cols
        When ``True``, the input columns are left in place alongside the
        components instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with the input columns replaced by the component columns.
        Backend (pandas/polars) is preserved end-to-end.
    """
    x = _frame.to_float_array(data.select(fit.columns))
    scores = (x - fit.mean) @ fit.rotation
    new_series = [
        nw.new_series(
            name=f"{prefix}{i + 1}",
            values=scores[:, i],
            backend=data.implementation,
        )
        for i in range(fit.num_comp)
    ]
    if not keep_original_cols:
        data = data.drop(fit.columns)
    return data.with_columns(*new_series)


class StepPls(Step):
    """Project numeric columns onto partial-least-squares components.

    The selected numeric columns are replaced by ``num_comp`` component
    columns ``{prefix}1 ... {prefix}{num_comp}``, or kept alongside them when
    ``keep_original_cols`` is ``True``. Each component is a linear
    combination of all the input columns chosen to covary with ``target``,
    so no single component belongs to any one of the input columns.
    Unlike unsupervised extraction such as `StepPca`, the components depend
    on the target as well as the selected columns. Asking for more
    components than there are input columns yields fewer.

    ``target`` may be numeric (partial least squares regression) or
    categorical/string (partial least squares discriminant analysis, fit by
    dummy-coding the classes into one indicator column per class).

    The columns are centered but not scaled, so a column measured on a
    larger numeric range contributes more to the components; normalize
    first if that is not what you want.

    Rows with a missing value in a selected column or a missing target are
    excluded when the components are fitted, and a row with a missing
    selected-column value gets ``NaN`` for every component at transform
    time.

    See [Arguments shared across steps](../user-guide/common-step-arguments.qmd)
    for the arguments this step shares with the others.

    Parameters
    ----------
    columns
        Columns to project. Accepts a single column name, a list of names, a
        mlfeatures selector such as ``Selectors.numeric_features()`` (see
        `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric
        columns are meaningful here.
    target
        Name of the column in the training frame that holds the numeric or
        categorical target. Read at fit time only; the column is *not*
        declared as an input feature in `schema_in()`.
    num_comp
        Number of components to keep. Must be a positive integer. Default
        ``5``.
    prefix
        Prefix for the produced component column names. Default ``"PLS"``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the components
        instead of being dropped. Default ``False``.
    role
        Role assigned to the new component columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"pls"``; a second step of this
        type in the same recipe becomes ``"pls_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    The two feature columns are replaced by one PLS component predictive of
    ``target``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepPls
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.1, 3.9, 6.1, 8.0],
            "target": [1.0, 2.0, 3.0, 4.0],
        }
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(
        StepPls(s.numeric_features() - "target", target="target", num_comp=1)
    )
    recipe
    ```

    `fit()` learns the component rotation:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the columns with their scores:

    ```{python}
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source columns alongside the components:

    ```{python}
    recipe = Recipe().add(
        StepPls(
            s.numeric_features() - "target",
            target="target",
            num_comp=1,
            keep_original_cols=True,
        )
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    A categorical ``target`` runs partial least squares discriminant
    analysis instead, by dummy-coding its classes:

    ```{python}
    df_class = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.1, 3.9, 6.1, 8.0],
            "target": ["a", "a", "b", "b"],
        }
    )
    recipe = Recipe().add(
        StepPls(s.numeric_features() - "target", target="target", num_comp=1)
    )
    fitted = recipe.fit(df_class)
    fitted.transform(df_class)
    ```
    """

    mode = Mode.MANY_TO_MANY

    #: Populated by ``fit()``: the input column order, centering mean, and rotation.
    _estimates: PlsFit | None

    def __init__(
        self,
        columns: Selector,
        target: str,
        *,
        num_comp: int = 5,
        prefix: str = "PLS",
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.target = target
        self.num_comp = num_comp
        self.prefix = prefix
        super().__init__(
            id, "pls", keep_original_cols=keep_original_cols, output_role=role
        )

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the component basis from the selected columns and target."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.lineage, info.producing_steps
        )
        target_series = training_data[self.target]
        self._estimates = fit_pls(selected, target_series, self.num_comp)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        comp_names = [f"{self.prefix}{i + 1}" for i in range(self._estimates.num_comp)]
        self._schema_out = ColSchema(
            columns=comp_names, dtypes={name: "Float64" for name in comp_names}
        )
        # Every component descends from the whole input set. The target
        # drove the fit but, matching StepClassDistance, is not treated as an
        # input feature the step transforms.
        self._output_parents = {name: list(selected.columns) for name in comp_names}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Project the selected columns onto the fitted components."""
        if self._estimates is None:
            raise RuntimeError("StepPls.transform() called before fit()")
        return apply_pls(
            new_data, self._estimates, self.prefix, self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepPls.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the component columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepPls.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each component descends from the whole set of input columns."""
        if self._output_parents is None:
            raise RuntimeError("StepPls.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """PLS is only meaningful on numeric columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        # The source columns are dropped; strip their roles so they do not
        # linger in the live role map. When they are kept, their roles stand.
        if self._schema_in is not None and not self.keep_original_cols:
            for col in self._schema_in.columns:
                for role in list(roles.of(col)):
                    roles.remove(col, role)
        if self._output_parents is not None and self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)

    def required_pkgs(self) -> list[str]:
        """PLS is computed with scikit-learn."""
        return ["numpy", "scikit-learn"]

    @property
    def produces_sparse(self) -> bool:
        """PLS components are dense by construction."""
        return False
