"""mlfeatures: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from mlfeatures.compat import RecipeTransformer
from mlfeatures.lineage import ColumnNode, Lineage
from mlfeatures.recipe import FittedRecipe, Recipe
from mlfeatures.schema import Schema
from mlfeatures.selectors import FeaturesSelector, Selectors
from mlfeatures.serialization import (
    from_dict,
    from_json,
    from_yaml,
    minimize,
    to_dict,
    to_json,
    to_yaml,
)
from mlfeatures.step import RowOp, Step, StepError
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
from mlfeatures.step_other import StepOther
from mlfeatures.step_pca import StepPca
from mlfeatures.step_pls import StepPls
from mlfeatures.step_polynomial import StepPolynomial
from mlfeatures.step_range import StepRange
from mlfeatures.step_spline_monotone import StepSplineMonotone
from mlfeatures.step_spline_natural import StepSplineNatural
from mlfeatures.step_unknown import StepUnknown
from mlfeatures.types import (
    ANY_DTYPE,
    NUMERIC_DTYPES,
    AcceptedDtypes,
    ColSchema,
    DataFrame,
    Frame,
    KeepOriginals,
    Mode,
    Roles,
    SchemaInfo,
    Selector,
)

try:
    __version__ = version("mlfeatures")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Recipe",
    "FittedRecipe",
    "StepClassDistance",
    "RecipeTransformer",
    "StepDateFeatures",
    "StepDatetimeFeatures",
    "StepDrop",
    "StepDropCorr",
    "StepDropNearZeroVariance",
    "StepDummy",
    "StepOneHot",
    "StepDropZeroVariance",
    "StepFilter",
    "StepGeoDistance",
    "StepHoliday",
    "StepMutate",
    "StepPca",
    "StepPls",
    "StepImputeMedian",
    "StepImputeMean",
    "StepImputeMode",
    "StepIndicateNa",
    "StepIntercept",
    "StepInteract",
    "StepLog",
    "StepNormalize",
    "StepOrderNorm",
    "StepOther",
    "StepPolynomial",
    "StepRange",
    "StepSplineMonotone",
    "StepSplineNatural",
    "StepUnknown",
    "Step",
    "RowOp",
    "StepError",
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "AcceptedDtypes",
    "Mode",
    "KeepOriginals",
    "Roles",
    "ColSchema",
    "SchemaInfo",
    "Selector",
    "Lineage",
    "ColumnNode",
    "Schema",
    "DataFrame",
    "Frame",
    "Selectors",
    "FeaturesSelector",
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
    "minimize",
]
