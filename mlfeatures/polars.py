"""Opt-in polars accessor: ``df.mlf.<step>(...)`` sugar over the mlfeatures lifecycle.

Importing this module registers a ``mlf`` namespace on polars DataFrames, so a
single step can be fitted and applied to a data frame in one expression:

```python
import mlfeatures.polars  # registers the `mlf` namespace (side effect)

df.mlf.step_dummy(["color"])                  # one step
df.mlf.apply(StepNormalize(["x"]))            # any step(s), generic
df.mlf.step_dummy(["color"]).mlf.step_normalize(["x"])  # chains
```

Each method fits the step on ``df`` and immediately transforms ``df``, returning a
fresh polars ``DataFrame`` (the fitted step is discarded). It is a thin convenience
layer over `Recipe`: it adds no new data-operation surface and
delegates conversion to the recipe lifecycle.

Boundary note: this is the **one** module in ``mlfeatures`` that imports ``polars`` at
runtime, a deliberate exception to the package's no-runtime-backend-import rule.
The accessor is an explicitly polars-only convenience and is kept out of the backend-agnostic core: it is never
imported by `mlfeatures.__init__`, so users opt in with ``import mlfeatures.polars``.
polars is already a hard dependency, so this adds no new requirement.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

from dateutil.rrule import rrule, rruleset

from mlfeatures._dependencies import import_optional
from mlfeatures.recipe import Recipe
from mlfeatures.step import Step
from mlfeatures.step_class_distance import StepClassDistance
from mlfeatures.step_date_features import StepDateFeatures
from mlfeatures.step_datetime_features import StepDatetimeFeatures
from mlfeatures.step_drop import StepDrop
from mlfeatures.step_drop_corr import StepDropCorr
from mlfeatures.step_drop_near_zero_variance import StepDropNearZeroVariance
from mlfeatures.step_drop_zero_variance import StepDropZeroVariance
from mlfeatures.step_dummy import StepDummy
from mlfeatures.step_filter import StepFilter
from mlfeatures.step_geo_distance import StepGeoDistance
from mlfeatures.step_holiday import StepHoliday
from mlfeatures.step_impute_mean import StepImputeMean
from mlfeatures.step_impute_median import StepImputeMedian
from mlfeatures.step_impute_mode import StepImputeMode
from mlfeatures.step_indicate_na import StepIndicateNa
from mlfeatures.step_interact import StepInteract
from mlfeatures.step_intercept import StepIntercept
from mlfeatures.step_log import StepLog
from mlfeatures.step_mutate import StepMutate
from mlfeatures.step_normalize import StepNormalize
from mlfeatures.step_one_hot import StepOneHot
from mlfeatures.step_order_norm import StepOrderNorm
from mlfeatures.step_other import DEFAULT_OTHER_LEVEL, StepOther
from mlfeatures.step_pca import StepPca
from mlfeatures.step_pls import StepPls
from mlfeatures.step_polynomial import StepPolynomial
from mlfeatures.step_range import StepRange
from mlfeatures.step_spline_monotone import StepSplineMonotone
from mlfeatures.step_spline_natural import StepSplineNatural
from mlfeatures.step_unknown import DEFAULT_UNKNOWN_LEVEL, StepUnknown

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Any, Literal

    import polars as pl

    from mlfeatures.step_class_distance import Metric
    from mlfeatures.step_geo_distance import GeoMetric
    from mlfeatures.types import Frame, Selector
else:
    # Runtime: this opt-in module is the one sanctioned place that imports polars
    # at runtime. Route it through
    # import_optional so a missing polars surfaces as a friendly, greppable
    # MissingDependencyError at ``import mlfeatures.polars`` rather than a bare error.
    pl = import_optional("polars", purpose="for the `df.mlf` accessor")

__all__ = ["MlfNamespace"]


@pl.api.register_dataframe_namespace("mlf")
class MlfNamespace:
    """The ``mlf`` polars DataFrame namespace.

    Registered on import; reach it as ``df.mlf``. Each method builds the
    corresponding step, fits it on the data frame, and returns the transformed data frame.
    """

    def __init__(self, df: pl.DataFrame) -> None:
        self._df = df

    def apply(self, *steps: Step) -> pl.DataFrame:
        """Fit ``steps`` on the data frame in order and return the transformed data frame.

        The generic escape hatch behind every ``step_*`` method: it accepts any
        `Step` instance(s), so steps without a dedicated
        wrapper (or custom steps) are reachable too.
        """
        fitted = Recipe(steps=list(steps)).fit(self._df)
        return cast("pl.DataFrame", fitted.transform(self._df))

    def step_dummy(
        self,
        columns: Selector,
        *,
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        role: str | None = None,
    ) -> pl.DataFrame:
        """Treatment (drop-one) dummy encoding. See `StepDummy`."""
        return self.apply(
            StepDummy(
                columns,
                name_fn=name_fn,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_one_hot(
        self,
        columns: Selector,
        *,
        name_fn: Callable[[str, str], str] | None = None,
        keep_original_cols: bool = False,
        role: str | None = None,
    ) -> pl.DataFrame:
        """One-hot encoding. See `StepOneHot`."""
        return self.apply(
            StepOneHot(
                columns,
                name_fn=name_fn,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_date_features(
        self,
        columns: Selector,
        *,
        features: list[str] | tuple[str, ...] = ("year", "month", "day", "dow", "doy"),
        categorical: list[str] | tuple[str, ...] = (),
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Expand Date columns into calendar-part feature columns. See `StepDateFeatures`."""
        return self.apply(
            StepDateFeatures(
                columns,
                features=features,
                categorical=categorical,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_datetime_features(
        self,
        columns: Selector,
        *,
        features: list[str] | tuple[str, ...] = (
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dow",
            "doy",
        ),
        categorical: list[str] | tuple[str, ...] = (),
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Expand Datetime columns into calendar-part feature columns. See `StepDatetimeFeatures`."""
        return self.apply(
            StepDatetimeFeatures(
                columns,
                features=features,
                categorical=categorical,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_drop(self, columns: Selector) -> pl.DataFrame:
        """Drop the selected columns. See `StepDrop`."""
        return self.apply(StepDrop(columns))

    def step_drop_corr(
        self,
        columns: Selector,
        *,
        threshold: float = 0.9,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pl.DataFrame:
        """Drop highly correlated columns. See `StepDropCorr`."""
        return self.apply(StepDropCorr(columns, threshold=threshold, method=method))

    def step_drop_near_zero_variance(
        self,
        columns: Selector,
        *,
        freq_cut: float = 95 / 5,
        unique_cut: float = 10,
    ) -> pl.DataFrame:
        """Drop sparse, unbalanced columns. See `StepDropNearZeroVariance`."""
        return self.apply(
            StepDropNearZeroVariance(columns, freq_cut=freq_cut, unique_cut=unique_cut)
        )

    def step_drop_zero_variance(self, columns: Selector) -> pl.DataFrame:
        """Drop zero-variance columns. See `StepDropZeroVariance`."""
        return self.apply(StepDropZeroVariance(columns))

    def step_intercept(
        self,
        *,
        value: float = 1.0,
        name: str = "intercept",
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Add a constant intercept column. See `StepIntercept`."""
        return self.apply(StepIntercept(value=value, name=name, role=role))

    def step_normalize(self, columns: Selector) -> pl.DataFrame:
        """Center and scale numeric columns. See `StepNormalize`."""
        return self.apply(StepNormalize(columns))

    def step_order_norm(self, columns: Selector) -> pl.DataFrame:
        """Normalize numeric columns onto a standard normal by rank. See `StepOrderNorm`."""
        return self.apply(StepOrderNorm(columns))

    def step_interact(
        self,
        columns: Selector,
        *,
        sep: str = "_x_",
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Add the pairwise product of every pair of selected columns. See `StepInteract`."""
        return self.apply(
            StepInteract(
                columns, sep=sep, keep_original_cols=keep_original_cols, role=role
            )
        )

    def step_range(
        self,
        columns: Selector,
        *,
        min: float = 0.0,
        max: float = 1.0,
        clipping: bool = True,
    ) -> pl.DataFrame:
        """Min-max scale numeric columns to a fixed range. See `StepRange`."""
        return self.apply(StepRange(columns, min=min, max=max, clipping=clipping))

    def step_log(
        self,
        columns: Selector,
        *,
        base: float = math.e,
        offset: float = 0.0,
        signed: bool = False,
    ) -> pl.DataFrame:
        """Take the logarithm of numeric columns. See `StepLog`."""
        return self.apply(StepLog(columns, base=base, offset=offset, signed=signed))

    def step_filter(
        self,
        predicate: Callable[[Frame], Any],
    ) -> pl.DataFrame:
        """Keep only the rows matching a predicate. See `StepFilter`."""
        return self.apply(StepFilter(predicate))

    def step_mutate(
        self,
        mutations: Mapping[str, Callable[[Frame], Any]],
        *,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Apply arbitrary column expressions. See `StepMutate`."""
        return self.apply(StepMutate(mutations, role=role))

    def step_pca(
        self,
        columns: Selector,
        *,
        num_comp: int = 5,
        prefix: str = "PC",
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Project numeric columns onto principal components. See `StepPca`."""
        return self.apply(
            StepPca(
                columns,
                num_comp=num_comp,
                prefix=prefix,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_pls(
        self,
        columns: Selector,
        target: str,
        *,
        num_comp: int = 5,
        prefix: str = "PLS",
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Project numeric columns onto PLS components. See `StepPls`."""
        return self.apply(
            StepPls(
                columns,
                target,
                num_comp=num_comp,
                prefix=prefix,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_class_distance(
        self,
        columns: Selector,
        target: str,
        *,
        metric: Metric = "euclidean",
        log_distance: bool = True,
        eps: float = 0.001,
        prefix: str = "distance",
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Per-class distance features. See `StepClassDistance`."""
        return self.apply(
            StepClassDistance(
                columns,
                target,
                metric=metric,
                log_distance=log_distance,
                eps=eps,
                prefix=prefix,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_geo_distance(
        self,
        lat: str,
        lon: str,
        *,
        ref_lat: float,
        ref_lon: float,
        metric: GeoMetric = "haversine",
        log_distance: bool = False,
        name: str = "geo_dist",
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Distance from each row to a fixed reference point. See `StepGeoDistance`."""
        return self.apply(
            StepGeoDistance(
                lat,
                lon,
                ref_lat=ref_lat,
                ref_lon=ref_lon,
                metric=metric,
                log_distance=log_distance,
                name=name,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_holiday(
        self,
        columns: Selector,
        *,
        holidays: list[str] | tuple[str, ...] = (
            "New Year's Day",
            "Labor Day",
            "Christmas Day",
        ),
        rules: Mapping[str, rrule | rruleset] | None = None,
        country: str = "US",
        subdiv: str | None = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Add binary indicator columns flagging named holidays and custom recurrences. See `StepHoliday`."""
        return self.apply(
            StepHoliday(
                columns,
                holidays=holidays,
                rules=rules,
                country=country,
                subdiv=subdiv,
                keep_original_cols=keep_original_cols,
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

    def step_indicate_na(
        self,
        columns: Selector,
        *,
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Add binary indicator columns flagging missing values. See `StepIndicateNa`."""
        return self.apply(
            StepIndicateNa(columns, keep_original_cols=keep_original_cols, role=role)
        )

    def step_polynomial(
        self,
        columns: Selector,
        *,
        degree: int = 2,
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Expand columns into a polynomial basis. See `StepPolynomial`."""
        return self.apply(
            StepPolynomial(
                columns,
                degree=degree,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_spline_natural(
        self,
        columns: Selector,
        *,
        deg_free: int = 10,
        complete_set: bool = False,
        knots: Sequence[float] | None = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Expand columns into a natural cubic spline basis. See `StepSplineNatural`."""
        return self.apply(
            StepSplineNatural(
                columns,
                deg_free=deg_free,
                complete_set=complete_set,
                knots=knots,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_spline_monotone(
        self,
        columns: Selector,
        *,
        deg_free: int = 10,
        degree: int = 3,
        complete_set: bool = True,
        knots: Sequence[float] | None = None,
        keep_original_cols: bool = False,
        role: str | None = "feature",
    ) -> pl.DataFrame:
        """Expand columns into a monotone (I-spline) basis. See `StepSplineMonotone`."""
        return self.apply(
            StepSplineMonotone(
                columns,
                deg_free=deg_free,
                degree=degree,
                complete_set=complete_set,
                knots=knots,
                keep_original_cols=keep_original_cols,
                role=role,
            )
        )

    def step_unknown(
        self,
        columns: Selector,
        *,
        new_level: str = DEFAULT_UNKNOWN_LEVEL,
    ) -> pl.DataFrame:
        """Recode missing categorical levels to an explicit level. See `StepUnknown`."""
        return self.apply(StepUnknown(columns, new_level=new_level))

    def step_other(
        self,
        columns: Selector,
        *,
        threshold: float | None = None,
        min_count: int | None = None,
        other_level: str = DEFAULT_OTHER_LEVEL,
    ) -> pl.DataFrame:
        """Collapse infrequent categorical levels. See `StepOther`."""
        return self.apply(
            StepOther(
                columns,
                threshold=threshold,
                min_count=min_count,
                other_level=other_level,
            )
        )
