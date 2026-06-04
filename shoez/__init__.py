"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.compat import RecipeTransformer
from shoez.recipe import FittedRecipe, Recipe
from shoez.selectors import Selectors
from shoez.step import RowOp, Step
from shoez.step_class_distance import (
    ClassDistanceFit,
    StepClassDistance,
    apply_class_distance,
    fit_class_distance,
)
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_dummy import StepDummy
from shoez.step_imputation_mean import StepImputationMean
from shoez.step_intercept import StepIntercept
from shoez.step_normalize import StepNormalize
from shoez.step_one_hot import StepOneHot
from shoez.step_polynomial import apply_polynomial, fit_polynomial
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
    "ClassDistanceFit",
    "StepClassDistance",
    "RecipeTransformer",
    "StepDropCorr",
    "StepDummy",
    "StepOneHot",
    "StepDropZeroVariance",
    "StepImputationMean",
    "StepIntercept",
    "StepNormalize",
    "StepUnknown",
    "Step",
    "RowOp",
    "fit_class_distance",
    "apply_class_distance",
    "fit_polynomial",
    "apply_polynomial",
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "Mode",
    "Roles",
    "ColSchema",
    "SchemaInfo",
    "DataFrame",
    "Frame",
    "Selectors",
]
