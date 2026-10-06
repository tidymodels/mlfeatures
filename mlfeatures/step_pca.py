"""Principal component analysis: fit + apply.

`step_pca` is the ``MANY_TO_MANY`` prototype from ``design/steps-catalog.md``,
modelled on ``step_pca`` from the R ``recipes`` package. A *set* of numeric input
columns is replaced by a new *set* of principal-component columns, so every
output descends from the whole input set rather than a single source column.

The *fit* half (`fit_pca()`) learns the centering mean and the component
loadings from the training data via an SVD; the *apply* half (`apply_pca()`)
reuses those statistics to project any frame onto the learned components — the
basis is never recomputed from new data, giving train/serve parity.
`StepPca` wraps both into the `Step` lifecycle so a `Recipe` can fit and apply
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import narwhals.stable.v2 as nw
import numpy as np

from mlfeatures import _frame
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

__all__ = ["PcaFit", "fit_pca", "apply_pca", "StepPca"]


@dataclass(frozen=True)
class PcaFit:
    """Principal-component basis learned at fit time.

    Bundles everything `apply_pca()` needs to project new data onto the same
    components the training set defined: the input column order, the centering
    mean, and the loadings matrix (columns are the principal axes). Marked
    ``frozen=True`` so the fitted state cannot be mutated and drift
    train-to-serve.

    Attributes
    ----------
    columns
        Input column names in the order the loadings rows correspond to.
    mean
        Training-set mean of each input column (length ``n_features``). Applied
        as ``X - mean`` before projection.
    loadings
        Matrix of shape ``(n_features, num_comp)``; column ``i`` is the ``i``-th
        principal axis. New scores are ``(X - mean) @ loadings``.
    num_comp
        Number of components actually produced (``loadings.shape[1]``), capped
        at the number of input columns and the number of complete rows.
    """

    columns: list[str]
    mean: np.ndarray[Any, Any]
    loadings: np.ndarray[Any, Any] = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    num_comp: int = 0


def fit_pca(selected: Frame, num_comp: int) -> PcaFit:
    """Learn the PCA centering mean and component loadings from ``selected``.

    Rows with any missing value are dropped before the SVD, so the basis is
    estimated on complete cases only. The number of components is capped at the
    number of input columns and the number of complete rows. The cap is a shape,
    not the numeric rank: on rank-deficient data the trailing components are
    retained and carry only numerical noise.

    Parameters
    ----------
    selected
        Internal `Frame` (narwhals) holding the numeric columns to project.
        Native frames are wrapped via `mlfeatures._frame.from_native()` at the
        recipe boundary.
    num_comp
        Number of principal components to keep. Must be an ``int`` strictly
        greater than zero. Booleans are rejected even though ``bool`` is an
        ``int`` subclass in Python — ``num_comp=True`` is almost always a bug.

    Returns
    -------
    PcaFit
        The fitted basis. Pair with `apply_pca()` to project a frame.

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
    # Complete cases only: a row with any NaN cannot contribute to the mean or
    # the SVD without contaminating them.
    complete = x[~np.isnan(x).any(axis=1)]
    mean = complete.mean(axis=0)
    xc = complete - mean
    # full_matrices=False keeps Vt at shape (rank, n_features); its rows are the
    # principal axes. Cap the kept components at the number of rows returned.
    # The numpy 2.2 stubs (the newest available on Python 3.10) leave the
    # `svd` result partially unknown to pyright; the cast below pins it.
    _, _, vt_raw = np.linalg.svd(xc, full_matrices=False)  # pyright: ignore[reportUnknownVariableType]
    vt = cast("np.ndarray[Any, Any]", vt_raw)
    eff = min(num_comp, vt.shape[0])
    top = cast("np.ndarray[Any, Any]", vt[:eff])
    loadings = cast("np.ndarray[Any, Any]", np.transpose(top))  # (n_features, eff)
    return PcaFit(columns=columns, mean=mean, loadings=loadings, num_comp=eff)


def apply_pca(
    data: Frame, fit: PcaFit, prefix: str, keep_original_cols: bool = False
) -> Frame:
    """Replace the fitted input columns with their principal-component scores.

    The input columns named in ``fit`` are dropped (unless ``keep_original_cols``)
    and ``fit.num_comp`` score columns named ``{prefix}1 ... {prefix}{num_comp}``
    are appended, computed as ``(X - fit.mean) @ fit.loadings``. Rows with a
    missing input value yield ``NaN`` scores. The basis comes from ``fit`` — it
    is not recomputed from ``data``.

    Parameters
    ----------
    data
        Internal `Frame` to transform. Must contain the columns in ``fit``.
    fit
        Output of `fit_pca()`: the centering mean and loadings.
    prefix
        Prefix for the produced component column names.
    keep_original_cols
        When ``True``, the input columns are left in place alongside the
        components instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with the input columns replaced by the component columns
        (appended at the end). Backend (pandas/polars) is preserved end-to-end.
    """
    x = _frame.to_float_array(data.select(fit.columns))
    scores = (x - fit.mean) @ fit.loadings
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


class StepPca(Step):
    """Project numeric columns onto their principal components.

    The selected numeric columns are replaced by ``num_comp``
    principal-component columns ``{prefix}1 ... {prefix}{num_comp}``, or kept
    alongside them when ``keep_original_cols`` is ``True``. Each
    component is a linear combination of all the input columns, so no single
    component belongs to any one of them. Asking for more components than there
    are input columns yields fewer.

    The columns are centered but not scaled, so a column measured on a larger
    numeric range contributes more to the components; normalize first if that is
    not what you want.

    Rows with a missing value are excluded when the components are computed, and
    such a row gets ``NaN`` for every component.

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
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    num_comp
        Number of principal components to keep. Must be a positive integer.
        Default ``5``.
    prefix
        Prefix for the produced component column names. Default ``"PC"``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the components
        instead of being dropped. Default ``False``.
    role
        Role assigned to the new component columns (see `Roles`). Default
        ``"feature"``.
    id
        Stable step identifier. Defaults to ``"pca"``; a second step of this
        type in the same recipe becomes ``"pca_2"``, and passing an ``id``
        already used in the recipe is an error.

    Examples
    --------
    The two correlated columns are replaced by two principal components:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from mlfeatures import Recipe, StepPca
    from mlfeatures import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0, 4.0], "y": [2.1, 3.9, 6.1, 8.0]}
    )
    df
    ```

    Specifying the step touches no data:

    ```{python}
    recipe = Recipe().add(StepPca(s.numeric_features(), num_comp=2))
    recipe
    ```

    `fit()` learns the component loadings:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` replaces the columns with their scores:

    ```{python}
    fitted.transform(df)
    ```

    ``num_comp`` caps how many components are kept:

    ```{python}
    recipe = Recipe().add(StepPca(s.numeric_features(), num_comp=1))
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source columns alongside the components:

    ```{python}
    recipe = Recipe().add(
        StepPca(s.numeric_features(), num_comp=2, keep_original_cols=True)
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.MANY_TO_MANY

    def __init__(
        self,
        columns: Selector,
        *,
        num_comp: int = 5,
        prefix: str = "PC",
        keep_original_cols: bool = False,
        role: str | None = "feature",
        id: str | None = None,
    ) -> None:
        self.columns = columns
        self.num_comp = num_comp
        self.prefix = prefix
        super().__init__(
            id, "pca", keep_original_cols=keep_original_cols, output_role=role
        )
        # Populated by ``fit()``.
        self._estimates: PcaFit | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Learn the component basis from the selected columns on training data."""
        selected = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        self._estimates = fit_pca(selected, self.num_comp)
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(selected).items()}
        self._schema_in = ColSchema(columns=selected.columns, dtypes=dtypes_in)
        comp_names = [f"{self.prefix}{i + 1}" for i in range(self._estimates.num_comp)]
        self._schema_out = ColSchema(
            columns=comp_names, dtypes={name: "Float64" for name in comp_names}
        )
        # Every component descends from the whole input set.
        self._output_parents = {name: list(selected.columns) for name in comp_names}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Project the selected columns onto the fitted components."""
        if self._estimates is None:
            raise RuntimeError("StepPca.transform() called before fit()")
        return apply_pca(
            new_data, self._estimates, self.prefix, self.keep_original_cols
        )

    def schema_in(self) -> ColSchema:
        """Schema of the columns this step consumed. Known after ``fit()``."""
        if self._schema_in is None:
            raise RuntimeError("StepPca.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the component columns this step produced."""
        if self._schema_out is None:
            raise RuntimeError("StepPca.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each component descends from the whole set of input columns."""
        if self._output_parents is None:
            raise RuntimeError("StepPca.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """PCA is only meaningful on numeric columns."""
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
        """PCA is computed with NumPy."""
        return ["numpy"]

    @property
    def produces_sparse(self) -> bool:
        """Principal components are dense by construction."""
        return False
