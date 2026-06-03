"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

from shoez.apply_shared import apply_filter
from shoez.step import RowOp, Step
from shoez.step_zero_variance import fit_zero_variance
from shoez.types import ANY_DTYPE, ColSchema, DataFrame, Mode, SchemaInfo

try:
    __version__ = version("shoez")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "Step",
    "RowOp",
    "ANY_DTYPE",
    "Mode",
    "ColSchema",
    "SchemaInfo",
    "DataFrame",
    "fit_zero_variance",
    "apply_filter",
]
