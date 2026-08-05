"""Class-distance feature: distance from each row to per-class centroids.

The *fit* half (`fit_class_distance()`) learns one multidimensional centroid per class
from a labelled training set. The *apply* half (`apply_class_distance()`)
replaces the selected columns with the distance from each row to each centroid.
`StepClassDistance` wraps them into the
`Step` lifecycle so a `Recipe` can
fit and apply it.

See `shoez.step_normalize` and `shoez.step_polynomial` for the
canonical examples of the same fit/apply pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import narwhals.stable.v2 as nw
import numpy as np
import numpy.typing as npt

from shoez import _frame
from shoez.step import Step
from shoez.types import (
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

__all__ = [
    "ClassDistanceFit",
    "Metric",
    "StepClassDistance",
    "apply_class_distance",
    "fit_class_distance",
]


# ``Literal`` makes the allowed metric values part of the type so a typo at the
# call site is caught by mypy/pyright. Add a new metric here when ready to
# support it — the runtime dispatch in ``_compute_distances`` and the relevant
# tests are what need to change in lockstep.
Metric = Literal["euclidean", "manhattan"]


@dataclass(frozen=True)
class ClassDistanceFit:
    """The fitted result returned by `fit_class_distance()`.

    Bundles everything `apply_class_distance()` needs to compute distances:
    the centroid matrix, the names and order of the predictor columns and class
    labels, plus the distance configuration so the apply call can be a one-arg
    invocation (no risk of mismatching fit-time and apply-time settings).

    Marked ``frozen=True`` so the fitted object is immutable — accidental
    mutation of a stored statistic is a common source of train/serve drift.

    Attributes
    ----------
    columns
        Predictor column names in the row order of `centroids`. Length
        equals the number of predictors ``p``.
    classes
        Class labels in the column order of `centroids`, sorted by
        `sorted()` so the order is deterministic across fits.
        Length equals the number of classes ``c``.
    centroids
        Float-64 matrix of shape ``(p, c)`` — row ``i`` is the centroid
        coordinate for predictor ``columns[i]``, column ``j`` is the centroid
        for class ``classes[j]``. This matches the orientation
        ``p predictors × c classes`` requested by the original spec.
    metric
        Distance function to apply: ``"euclidean"`` or ``"manhattan"``.
    log_distance
        Whether `apply_class_distance()` should return
        ``log(distance + eps)`` rather than the raw distance.
    eps
        Small positive offset added inside the log so ``log(0)`` (a row sitting
        exactly on a centroid) does not produce ``-inf``. Unused when
        ``log_distance`` is ``False``.
    """

    columns: list[str]
    classes: list[Any]
    centroids: npt.NDArray[np.float64]
    metric: Metric
    log_distance: bool
    eps: float


def fit_class_distance(
    features: Frame,
    outcome: nw.Series[Any],
    *,
    metric: Metric = "euclidean",
    log_distance: bool = True,
    eps: float = 0.001,
) -> ClassDistanceFit:
    """Estimate per-class centroids from labelled training data.

    For each unique class label, the centroid is the column-wise mean of the
    rows belonging to that class. The fitted object pairs the centroid matrix
    with the column names, class labels, and distance configuration so
    `apply_class_distance()` can reconstruct everything without extra args.

    Parameters
    ----------
    features
        Narwhals ``DataFrame`` of numeric predictor columns. Dtype screening is
        the selector / ``accepts_dtypes`` layer's job; this function does not
        re-validate it (matching `fit_normalize()`).
    outcome
        Narwhals ``Series`` of class labels, aligned row-for-row with
        ``features``. Rows where the outcome is null are dropped before the
        per-class means are taken — a missing label is not a class.
    metric
        Distance function to use at apply time. Recorded on the fit object;
        does not change the centroid computation itself (Euclidean and
        Manhattan both use the arithmetic mean as their natural centroid).
    log_distance
        If ``True`` (the default), `apply_class_distance()` returns
        ``log(distance + eps)``. The log compresses the long tail of raw
        distances, which is usually what downstream models want.
    eps
        Small positive offset added before taking the log. The default
        ``0.001``.

    Returns
    -------
    ClassDistanceFit
        Fitted object containing the centroid matrix and configuration. See
        `ClassDistanceFit` for the field-by-field layout.

    The returned `ClassDistanceFit` holds one centroid per class in ``classes``
    order; its ``centroids`` matrix has shape ``(n_features, n_classes)``. See
    `StepClassDistance` for a runnable end-to-end example.
    """
    # Empty feature selections never reach here: the recipe detects a
    # zero-column selection and treats the step as the identity transform, so
    # ``fit`` is not called. See ``Recipe.fit`` / ``Step._empty_selection``.
    #
    # Drop rows where the outcome is null: a null label isn't a class, so it
    # can't contribute to any centroid. Filtering on the outcome side keeps the
    # rest of the function dealing only with rows that have a real label.
    # We build a single-column frame so the standard ``drop_nulls()`` semantics
    # apply, then read back the filtered indices via a boolean mask.
    outcome_np = outcome.to_numpy()
    # ``pd.isna`` would also work here; ``np.equal`` against itself catches NaN
    # (NaN != NaN) without importing pandas. For object arrays (e.g. strings),
    # ``np.equal(x, x)`` returns ``True`` for non-None and ``False`` for None.
    # We treat None and NaN as "missing".
    not_null = np.array([x is not None and not _is_nan(x) for x in outcome_np])

    # Pull the predictor columns into a (n, p) float array. ``to_numpy()`` on a
    # narwhals frame returns an ndarray with the columns in ``features.columns``
    # order, so the row index of the resulting centroid matrix maps 1-to-1 to
    # that list.
    x = _frame.to_float_array(features)
    # Apply the null-outcome mask to both the labels and the predictors so they
    # stay aligned row-for-row.
    x = x[not_null]
    labels = outcome_np[not_null]

    # Sort the unique class labels so the column order of ``centroids`` is
    # deterministic — same input always gives the same column layout. This
    # matches the ``sorted(levels)`` discipline used by ``fit_dummy``.
    classes: list[Any] = sorted(set(labels.tolist()))

    p = x.shape[1]
    c = len(classes)
    # Pre-allocate the (p, c) result: row ``i`` will hold predictor ``i``'s
    # mean across classes; column ``j`` will hold class ``j``'s centroid.
    centroids = np.empty((p, c), dtype=np.float64)
    for j, cls in enumerate(classes):
        # Boolean mask over rows: True where the label equals this class.
        class_mask = labels == cls
        # Per-column mean across the matching rows — that's the centroid.
        # ``axis=0`` collapses the row axis, yielding a length-``p`` vector.
        # ``np.mean`` of an empty array would produce a warning and a NaN; we
        # are guaranteed at least one row per class because each ``cls`` came
        # from ``set(labels)``, so empty groups aren't possible here.
        centroids[:, j] = np.mean(x[class_mask], axis=0)

    return ClassDistanceFit(
        columns=list(features.columns),
        classes=classes,
        centroids=centroids,
        metric=metric,
        log_distance=log_distance,
        eps=eps,
    )


def apply_class_distance(
    data: Frame,
    fit_result: ClassDistanceFit,
    *,
    prefix: str = "distance",
    keep_original_cols: bool = False,
) -> Frame:
    """Replace the fitted input columns with per-row distances to each centroid.

    For every class ``k`` in ``fit_result.classes``, computes the distance
    from each row of ``data`` (restricted to ``fit_result.columns``) to the
    centroid for ``k`` and produces a column named ``f"{prefix}_{k}"``. The
    input columns named in ``fit_result`` are dropped unless
    ``keep_original_cols``; every other column of ``data`` passes through
    unchanged.

    Parameters
    ----------
    data
        Narwhals ``DataFrame`` whose columns include those named in
        ``fit_result.columns``. Extra columns are passed through unchanged.
    fit_result
        Output of `fit_class_distance()`. Carries the centroid matrix and
        the distance configuration (metric, log toggle, epsilon).
    prefix
        Prefix for the new column names. Default ``"distance"`` produces
        ``distance_<class>``; override with e.g. ``"dist"`` for shorter names.
    keep_original_cols
        When ``True``, the input columns are left in place alongside the
        distance columns instead of being dropped.

    Returns
    -------
    Frame
        ``data`` with the input columns replaced by ``len(fit_result.classes)``
        new ``Float64`` columns. The row count is preserved.

    Notes
    -----
    NaN propagation: a null in any predictor column produces ``NaN`` for that
    row's distances in every new column. NumPy's arithmetic handles this
    naturally; other rows are unaffected (parallel to
    `apply_polynomial()`'s per-row null tolerance).

    The column name is the same whether or not the distances were
    log-transformed — the transformation is recorded in ``fit_result.log_distance``
    rather than in the column name.
    """
    # Empty fit (no features were selected) — nothing to add; pass through.
    if not fit_result.columns:
        return data

    # Pull the predictor columns out as a (n, p) float array. ``select`` keeps
    # the order we stored at fit time so the row index of ``X`` matches the
    # row index of ``fit_result.centroids``.
    x = _frame.to_float_array(_frame.select(data, fit_result.columns))

    # All the per-metric math lives in this one helper so the dispatch is in
    # one place and easy to extend.
    distances = _compute_distances(x, fit_result.centroids, fit_result.metric)

    # Optional log transform. ``+ eps`` shifts the distances strictly positive
    # before the log, so a row sitting exactly on a centroid (distance 0)
    # becomes ``log(eps)`` rather than ``-inf``.
    if fit_result.log_distance:
        distances = np.log(distances + fit_result.eps)

    # Attach the new distance columns via ``with_columns`` rather than
    # ``nw.concat(how="horizontal")``: on pandas, horizontal concat aligns by
    # *index*, which silently produces NaN when ``data`` has a non-default
    # index (e.g. after ``sklearn.train_test_split``). ``with_columns`` on a
    # series built with the same backend is positional, preserving the input
    # frame's index.
    new_series = [
        nw.new_series(
            name=f"{prefix}_{cls}",
            values=distances[:, j],
            backend=data.implementation,
        )
        for j, cls in enumerate(fit_result.classes)
    ]
    if not keep_original_cols:
        data = data.drop(fit_result.columns)
    return data.with_columns(*new_series)


# --- Private helpers --------------------------------------------------------


def _compute_distances(
    x: npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
    metric: Metric,
) -> npt.NDArray[np.float64]:
    """Distances from each row of ``x`` to each class centroid.

    Parameters
    ----------
    x
        Predictor matrix of shape ``(n, p)``.
    centroids
        Centroid matrix of shape ``(p, c)`` — same layout as
        `ClassDistanceFit.centroids`.
    metric
        Which distance function to compute.

    Returns
    -------
    ndarray
        Float-64 matrix of shape ``(n, c)``; entry ``(i, j)`` is the distance
        from row ``i`` of ``x`` to the centroid for class ``j``.

    Notes
    -----
    Implementation uses broadcasting to compute all pairwise differences in
    one expression:

    * ``x[:, :, None]`` reshapes ``x`` to ``(n, p, 1)``.
    * ``centroids[None, :, :]`` reshapes the centroid matrix to ``(1, p, c)``.
    * Subtracting them broadcasts to ``(n, p, c)`` — for every (row, predictor,
      class) triple, the difference between that row's predictor value and the
      corresponding centroid coordinate.

    From there each metric collapses the predictor axis (``axis=1``) in its
    own way: square + sum + sqrt for Euclidean, abs + sum for Manhattan.
    """
    # Compute the (n, p, c) difference tensor once and reuse it. For very large
    # n × p × c this is wasteful; with the small frames it's the clearest
    # expression of the math.
    diff = x[:, :, None] - centroids[None, :, :]

    # ``np.sqrt`` / ``np.sum`` lose their dtype in numpy's stubs (they're
    # overloaded across many input types), so we re-assert the concrete dtype
    # on the way out. ``np.asarray`` is a no-op when the input is already a
    # float64 ndarray, so this is just for the type-checker.
    if metric == "euclidean":
        result = np.sqrt(np.sum(diff * diff, axis=1))
        return np.asarray(result, dtype=np.float64)
    if metric == "manhattan":
        result = np.sum(np.abs(diff), axis=1)
        return np.asarray(result, dtype=np.float64)
    # The Literal type means mypy/pyright already catch typos, but a runtime
    # guard helps callers passing string literals from untyped sources (CLI,
    # YAML configs, etc.).
    raise ValueError(  # pragma: no cover
        f"unknown metric {metric!r}; expected 'euclidean' or 'manhattan'"
    )


def _is_nan(value: Any) -> bool:
    """``True`` for a floating-point NaN, and for any value unequal to itself."""
    # ``math.isnan`` raises on non-numeric types; pandas' ``isna`` would work
    # too but pulls in pandas as a runtime dep. The pattern ``v != v`` is the
    # classic NaN check (NaN is the only value not equal to itself).
    try:
        return bool(value != value)
    except TypeError:  # pragma: no cover
        return False


class StepClassDistance(Step):
    """Replace numeric columns with per-row distances to each class centroid.

    The selected columns are read to learn one centroid per outcome class, then
    replaced by one ``distance_<class>`` column per class holding each row's
    distance to that class's centroid, or kept alongside them when
    ``keep_original_cols`` is ``True``. Each distance is computed from all the
    selected columns at once, so no distance column belongs to any one of them.

    Parameters
    ----------
    columns
        Columns to measure distances over. Accepts a single column name, a list
        of names, a shoez selector such as ``Selectors.numeric_predictors()``
        (see `Selectors` for the full set), or a narwhals or polars selector
        (``narwhals.selectors.numeric()``, ``polars.selectors.numeric()``).
        Selectors are resolved against the frame's schema when the recipe is
        fitted, so the same selector works on any backend. Only numeric columns
        are meaningful here.
    outcome
        Name of the column in the training frame that holds the class labels.
        Read at fit time only; the column is *not* declared as an input
        feature in `schema_in()`.
    metric
        Distance function: ``"euclidean"`` (default) or ``"manhattan"``.
    log_distance
        Whether to return ``log(distance + eps)`` instead of raw distances.
        Default ``True``.
    eps
        Offset inside the log (only used when ``log_distance`` is ``True``).
        Default ``0.001``.
    prefix
        Prefix for the new column names. Default ``"distance"`` gives
        ``distance_<class>``.
    keep_original_cols
        If ``True``, the source columns are kept alongside the distance columns
        instead of being dropped. Default ``False``.
    role
        Role assigned to the new distance columns. Default ``"predictor"``.
    id
        Stable step identifier. Default ``"class_distance"``.

    Examples
    --------
    The two numeric columns are replaced by one ``distance_<class>`` column per
    class label found in ``outcome``:

    ```{python}
    #| echo: false
    #| output: false
    import pandas as pd
    from shoez import Recipe, StepClassDistance
    from shoez import Selectors as s
    ```

    ```{python}
    df = pd.DataFrame(
        {
            "x": [0.0, 1.0, 10.0, 11.0],
            "y": [0.0, 1.0, 10.0, 11.0],
            "label": ["a", "a", "b", "b"],
        }
    )
    df
    ```

    Specifying the step touches no data, so the recipe reports itself as
    unprepped:

    ```{python}
    recipe = Recipe().add(
        StepClassDistance(
            s.numeric_predictors(), outcome="label", log_distance=False
        )
    )
    recipe
    ```

    `fit()` learns one centroid per class label and returns a prepped recipe:

    ```{python}
    fitted = recipe.fit(df)
    fitted
    ```

    `transform()` then replaces the numeric columns with one
    ``distance_<class>`` column per label:

    ```{python}
    fitted.transform(df)
    ```

    ``log_distance`` (the default) compresses the raw distances:

    ```{python}
    recipe = Recipe().add(
        StepClassDistance(s.numeric_predictors(), outcome="label")
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```

    ``keep_original_cols`` keeps the source columns alongside the distances:

    ```{python}
    recipe = Recipe().add(
        StepClassDistance(
            s.numeric_predictors(), outcome="label", keep_original_cols=True
        )
    )
    fitted = recipe.fit(df)
    fitted.transform(df)
    ```
    """

    mode = Mode.MANY_TO_MANY

    def __init__(
        self,
        columns: Selector,
        outcome: str,
        *,
        metric: Metric = "euclidean",
        log_distance: bool = True,
        eps: float = 0.001,
        prefix: str = "distance",
        keep_original_cols: bool = False,
        role: str | None = "predictor",
        id: str | None = None,
    ):
        self.columns = columns
        self.outcome = outcome
        # Pin the attribute type to ``Metric`` so type-checkers don't widen the
        # stored value back to ``str`` and complain when we hand it to
        # ``fit_class_distance(metric=...)`` later.
        self.metric: Metric = metric
        self.log_distance = log_distance
        self.eps = eps
        self.prefix = prefix
        self.id = id if id is not None else "class_distance"
        self.trained = False
        self.keep_original_cols = keep_original_cols
        self.output_role = role
        # Populated by ``fit()`` — both schemas plus the fitted statistics.
        self._estimates: ClassDistanceFit | None = None
        self._schema_in: ColSchema | None = None
        self._schema_out: ColSchema | None = None
        self._output_parents: dict[str, list[str]] | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        """Estimate per-class centroids from the predictor + outcome columns."""
        features = _frame.select(
            training_data, self.columns, info.roles, info.schema, info.producing_steps
        )
        # ``training_data[col]`` returns a narwhals Series — the same shape
        # ``_ortho_poly`` consumes in step_polynomial.
        outcome_series = training_data[self.outcome]
        self._estimates = fit_class_distance(
            features,
            outcome_series,
            metric=self.metric,
            log_distance=self.log_distance,
            eps=self.eps,
        )
        # ``schema_in`` records only the predictor columns; the outcome is
        # consumed at fit time but isn't an input feature the step transforms.
        dtypes_in = {name: str(dt) for name, dt in _frame.dtypes(features).items()}
        self._schema_in = ColSchema(columns=features.columns, dtypes=dtypes_in)
        # ``schema_out`` lists the new distance columns; all are Float64.
        out_names = [f"{self.prefix}_{cls}" for cls in self._estimates.classes]
        self._schema_out = ColSchema(
            columns=out_names,
            dtypes=dict.fromkeys(out_names, "Float64"),
        )
        # Every distance column is derived from the full feature set (the
        # outcome drives centroid fitting but is not a transform-time input).
        self._output_parents = {name: list(features.columns) for name in out_names}
        return self

    def transform(self, new_data: Frame) -> Frame:
        """Append the distance columns to ``new_data``. Never re-estimates."""
        if self._estimates is None:
            raise RuntimeError("StepClassDistance.transform() called before fit()")
        return apply_class_distance(
            new_data,
            self._estimates,
            prefix=self.prefix,
            keep_original_cols=self.keep_original_cols,
        )

    def schema_in(self) -> ColSchema:
        """Schema of the predictor columns consumed at fit time."""
        if self._schema_in is None:
            raise RuntimeError("StepClassDistance.schema_in() called before fit()")
        return self._schema_in

    def schema_out(self) -> ColSchema:
        """Schema of the new distance columns produced by ``transform``."""
        if self._schema_out is None:
            raise RuntimeError("StepClassDistance.schema_out() called before fit()")
        return self._schema_out

    def output_parents(self) -> dict[str, list[str]]:
        """Each distance column is derived from the full feature set."""
        if self._output_parents is None:
            raise RuntimeError("StepClassDistance.output_parents() called before fit()")
        return self._output_parents

    def accepts_dtypes(self) -> AcceptedDtypes:
        """Distances are only meaningful on numeric predictor columns."""
        return set(NUMERIC_DTYPES)

    def update_roles(self, roles: Roles) -> None:
        # The source columns are dropped; strip their roles so they do not
        # linger in the live role map. The outcome column is untouched, and
        # when the sources are kept their roles stand.
        if self._schema_in is not None and not self.keep_original_cols:
            for col in self._schema_in.columns:
                for role in list(roles.of(col)):
                    roles.remove(col, role)
        if self._output_parents is not None and self.output_role is not None:
            for col in self._output_parents:
                roles.add(col, self.output_role)

    @property
    def produces_sparse(self) -> bool:
        """Distances are dense by construction — no zeros structurally."""
        return False
