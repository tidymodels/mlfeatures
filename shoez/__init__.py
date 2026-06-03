"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.recipe import FittedRecipe, Recipe
from shoez.step import RowOp, Step
from shoez.step_drop_corr import StepDropCorr
from shoez.step_drop_zero_variance import StepDropZeroVariance
from shoez.step_normalize import StepNormalize
from shoez.step_polynomial import apply_polynomial, fit_polynomial
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
    "StepDropCorr",
    "StepDropZeroVariance",
    "StepNormalize",
    "Step",
    "RowOp",
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
]
