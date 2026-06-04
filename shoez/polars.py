"""Opt-in polars accessor: ``df.shz.<step>(...)`` sugar over the shoez lifecycle.

Importing this module registers a ``shz`` namespace on polars DataFrames, so a
single step can be fitted and applied to a frame in one expression::

    import shoez.polars  # registers the `shz` namespace (side effect)

    df.shz.step_dummy(["color"])                  # one step
    df.shz.apply(StepNormalize(["x"]))            # any step(s), generic
    df.shz.step_dummy(["color"]).shz.step_normalize(["x"])  # chains

Each method fits the step on ``df`` and immediately transforms ``df``, returning a
fresh polars ``DataFrame`` (the fitted step is discarded). It is a thin convenience
layer over :class:`~shoez.recipe.Recipe` — it adds no new data-operation surface and
delegates conversion to the recipe lifecycle.

Boundary note: this is the **one** module in ``shoez`` that imports ``polars`` at
runtime, a deliberate exception to the package's no-runtime-backend-import rule
(``design/ecosystem.md``, boundary rule #5). The accessor is an explicitly
polars-only convenience and is kept out of the backend-agnostic core: it is never
imported by :mod:`shoez.__init__`, so users opt in with ``import shoez.polars``.
polars is already a hard dependency, so this adds no new requirement.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from shoez._dependencies import import_optional
from shoez.recipe import Recipe
from shoez.step import Step
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_intercept import StepIntercept
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    import polars as pl

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
        :class:`~shoez.step.Step` instance(s), so steps without a dedicated
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
        """Treatment (drop-one) dummy encoding. See :class:`~shoez.step_dummy.StepDummy`."""
        return self.apply(StepDummy(columns, name_fn, keep_original_cols))

    def step_one_hot(
        self,
        columns: list[str],
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
    ) -> pl.DataFrame:
        """One-hot encoding. See :class:`~shoez.step_one_hot.StepOneHot`."""
        return self.apply(StepOneHot(columns, name_fn, keep_original_cols))

    def step_drop_corr(
        self,
        columns: Selector,
        threshold: float = 0.9,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pl.DataFrame:
        """Drop highly correlated columns. See :class:`~shoez.step_drop_corr.StepDropCorr`."""
        return self.apply(StepDropCorr(columns, threshold, method))

    def step_drop_zero_variance(self, columns: list[str]) -> pl.DataFrame:
        """Drop zero-variance columns. See :class:`~shoez.step_drop_zero_variance.StepDropZeroVariance`."""
        return self.apply(StepDropZeroVariance(columns))

    def step_intercept(
        self,
        value: float = 1.0,
        name: str = "intercept",
        role: str | None = "predictor",
    ) -> pl.DataFrame:
        """Add a constant intercept column. See :class:`~shoez.step_intercept.StepIntercept`."""
        return self.apply(StepIntercept(value, name, role))

    def step_normalize(self, columns: list[str]) -> pl.DataFrame:
        """Center and scale numeric columns. See :class:`~shoez.step_normalize.StepNormalize`."""
        return self.apply(StepNormalize(columns))
