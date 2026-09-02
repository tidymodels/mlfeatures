"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.compat import RecipeTransformer
from shoez.ptype import Ptype
from shoez.recipe import FittedRecipe, Recipe
from shoez.schema import ColumnNode, Schema
from shoez.selectors import Selectors
from shoez.serialization import (
    from_dict,
    from_json,
    from_yaml,
    minimize,
    to_dict,
    to_json,
    to_yaml,
)
from shoez.step import RowOp, Step, StepError
from shoez.step_class_distance import StepClassDistance
from shoez.step_date_features import StepDateFeatures
from shoez.step_datetime_features import StepDatetimeFeatures
from shoez.step_drop import StepDrop
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_near_zero_variance import StepDropNearZeroVariance
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_filter import StepFilter
from shoez.step_geo_distance import StepGeoDistance
from shoez.step_impute_mean import StepImputeMean
from shoez.step_impute_median import StepImputeMedian
from shoez.step_impute_mode import StepImputeMode
from shoez.step_indicate_na import StepIndicateNa
from shoez.step_interact import StepInteract
from shoez.step_intercept import StepIntercept
from shoez.step_log import StepLog
from shoez.step_mutate import StepMutate
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot
from shoez.step_order_norm import StepOrderNorm
from shoez.step_other import StepOther
from shoez.step_pca import StepPca
from shoez.step_pls import StepPls
from shoez.step_polynomial import StepPolynomial
from shoez.step_range import StepRange
from shoez.step_unknown import StepUnknown
from shoez.types import (
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
    __version__ = version("shoez")
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
    "Schema",
    "ColumnNode",
    "Ptype",
    "DataFrame",
    "Frame",
    "Selectors",
    "to_dict",
    "from_dict",
    "to_json",
    "from_json",
    "to_yaml",
    "from_yaml",
    "minimize",
]
