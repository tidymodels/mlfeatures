"""Opt-in polars accessor: ``df.shz.<step>(...)`` sugar over the shoez lifecycle.

Importing this module registers a ``shz`` namespace on polars DataFrames, so a
single step can be fitted and applied to a frame in one expression:

```python
import shoez.polars  # registers the `shz` namespace (side effect)

df.shz.step_dummy(["color"])                  # one step
df.shz.apply(StepNormalize(["x"]))            # any step(s), generic
df.shz.step_dummy(["color"]).shz.step_normalize(["x"])  # chains
```

Each method fits the step on ``df`` and immediately transforms ``df``, returning a
fresh polars ``DataFrame`` (the fitted step is discarded). It is a thin convenience
layer over `Recipe` — it adds no new data-operation surface and
delegates conversion to the recipe lifecycle.

Boundary note: this is the **one** module in ``shoez`` that imports ``polars`` at
runtime, a deliberate exception to the package's no-runtime-backend-import rule
(``design/ecosystem.md``, boundary rule #5). The accessor is an explicitly
polars-only convenience and is kept out of the backend-agnostic core: it is never
imported by `shoez.__init__`, so users opt in with ``import shoez.polars``.
polars is already a hard dependency, so this adds no new requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from shoez._dependencies import import_optional
from shoez.recipe import Recipe
from shoez.step import Step
from shoez.step_class_distance import StepClassDistance
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_impute_mean import StepImputeMean
from shoez.step_impute_median import StepImputeMedian
from shoez.step_impute_mode import StepImputeMode
from shoez.step_intercept import StepIntercept
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot
from shoez.step_polynomial import StepPolynomial
from shoez.step_unknown import DEFAULT_UNKNOWN_LEVEL, StepUnknown

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import polars as pl

    from shoez.step_class_distance import Metric
    from shoez.types import Selector
else:
    # Runtime: this opt-in module is the one sanctioned place that imports polars
    # at runtime (``design/ecosystem.md`` boundary rule #5). Route it through
    # import_optional so a missing polars surfaces as a friendly, greppable
    # MissingDependencyError at ``import shoez.polars`` rather than a bare error.
    pl = import_optional("polars", purpose="for the `df.shz` accessor")

__all__ = ["ShzNamespace"]


@pl.api.register_dataframe_namespace("shz")
class ShzNamespace:
    """The ``shz`` polars DataFrame namespace.

    Registered on import; reach it as ``df.shz``. Each method builds the
    corresponding step, fits it on the frame, and returns the transformed frame.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def apply(self, *steps: Step) -> pl.DataFrame:
        """Fit ``steps`` on the frame in order and return the transformed frame.

        The generic escape hatch behind every ``step_*`` method: it accepts any
        `Step` instance(s), so steps without a dedicated
        wrapper (or custom steps) are reachable too.
        """
        fitted = Recipe(steps=list(steps)).fit(self._df)
        return cast("pl.DataFrame", fitted.transform(self._df))

    def step_dummy(
        self,
        columns: list[str],
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
    ) -> pl.DataFrame:
        """Treatment (drop-one) dummy encoding. See `StepDummy`."""
        return self.apply(StepDummy(columns, name_fn, keep_original_cols))

    def step_one_hot(
        self,
        columns: list[str],
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
    ) -> pl.DataFrame:
        """One-hot encoding. See `StepOneHot`."""
        return self.apply(StepOneHot(columns, name_fn, keep_original_cols))

    def step_drop_corr(
        self,
        columns: Selector,
        threshold: float = 0.9,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pl.DataFrame:
        """Drop highly correlated columns. See `StepDropCorr`."""
        return self.apply(StepDropCorr(columns, threshold, method))

    def step_drop_zero_variance(self, columns: list[str]) -> pl.DataFrame:
        """Drop zero-variance columns. See `StepDropZeroVariance`."""
        return self.apply(StepDropZeroVariance(columns))

    def step_intercept(
        self,
        value: float = 1.0,
        name: str = "intercept",
        role: str | None = "predictor",
    ) -> pl.DataFrame:
        """Add a constant intercept column. See `StepIntercept`."""
        return self.apply(StepIntercept(value, name, role))

    def step_normalize(self, columns: list[str]) -> pl.DataFrame:
        """Center and scale numeric columns. See `StepNormalize`."""
        return self.apply(StepNormalize(columns))

    def step_class_distance(
        self,
        columns: Selector,
        outcome: str,
        *,
        metric: Metric = "euclidean",
        log_distance: bool = True,
        eps: float = 0.001,
        prefix: str = "distance",
        role: str | None = "predictor",
    ) -> pl.DataFrame:
        """Per-class distance features. See `StepClassDistance`."""
        return self.apply(
            StepClassDistance(
                columns,
                outcome,
                metric=metric,
                log_distance=log_distance,
                eps=eps,
                prefix=prefix,
                role=role,
            )
        )

    def step_impute_mean(self, columns: Selector) -> pl.DataFrame:
        """Impute nulls with the column mean. See `StepImputeMean`."""
        return self.apply(StepImputeMean(columns))

    def step_impute_median(self, columns: Selector) -> pl.DataFrame:
        """Impute nulls with the column median. See `StepImputeMedian`."""
        return self.apply(StepImputeMedian(columns))

    def step_impute_mode(self, columns: Selector) -> pl.DataFrame:
        """Impute nulls with the column mode. See `StepImputeMode`."""
        return self.apply(StepImputeMode(columns))

    def step_polynomial(
        self,
        columns: Selector,
        *,
        degree: int = 2,
        role: str | None = "predictor",
    ) -> pl.DataFrame:
        """Expand columns into a polynomial basis. See `StepPolynomial`."""
        return self.apply(StepPolynomial(columns, degree=degree, role=role))

    def step_unknown(
        self,
        columns: Selector,
        new_level: str = DEFAULT_UNKNOWN_LEVEL,
    ) -> pl.DataFrame:
        """Recode unseen categorical levels. See `StepUnknown`."""
        return self.apply(StepUnknown(columns, new_level))
