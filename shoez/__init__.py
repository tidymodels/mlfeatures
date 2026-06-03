"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.recipe import FittedRecipe, Recipe
from shoez.step import RowOp, Step
from shoez.step_drop_corr import StepDropCorr
from shoez.types import (
    ANY_DTYPE,
    NUMERIC_DTYPES,
    ColSchema,
    DataFrame,
    Frame,
    Mode,
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
    "Step",
    "RowOp",
    "ANY_DTYPE",
    "NUMERIC_DTYPES",
    "Mode",
    "ColSchema",
    "SchemaInfo",
    "DataFrame",
    "Frame",
]
