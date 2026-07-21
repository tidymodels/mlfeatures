"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.compat import RecipeTransformer
from shoez.ptype import Ptype
from shoez.recipe import FittedRecipe, Recipe
from shoez.schema import ColumnNode, Schema
from shoez.selectors import Selectors
from shoez.serialization import from_dict, from_json, minimize, to_dict, to_json
from shoez.step import RowOp, Step, StepError
from shoez.step_class_distance import StepClassDistance
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_filter import StepFilter
from shoez.step_impute_mean import StepImputeMean
from shoez.step_impute_median import StepImputeMedian
from shoez.step_impute_mode import StepImputeMode
from shoez.step_intercept import StepIntercept
from shoez.step_mutate import StepMutate
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot
from shoez.step_pca import StepPca
from shoez.step_polynomial import StepPolynomial
from shoez.step_unknown import StepUnknown
from shoez.types import (
    ANY_DTYPE,
    NUMERIC_DTYPES,
    ColSchema,
    DataFrame,
    Frame,
    Mode,
    Roles,
    SchemaInfo,
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
    "StepDropCorr",
    "StepDummy",
    "StepOneHot",
    "StepDropZeroVariance",
    "StepFilter",
    "StepMutate",
    "StepPca",
    "StepImputeMedian",
    "StepImputeMean",
    "StepImputeMode",
    "StepIntercept",
    "StepNormalize",
    "StepPolynomial",
    "StepUnknown",
    "Step",
    "RowOp",
    "StepError",
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "Mode",
    "Roles",
    "ColSchema",
    "SchemaInfo",
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
    "minimize",
]
