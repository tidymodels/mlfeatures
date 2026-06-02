"""shoez: a framework for declaring, estimating, and applying feature-engineering pipelines."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("shoez")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = ["__version__"]
