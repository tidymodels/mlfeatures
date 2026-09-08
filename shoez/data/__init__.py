"""Example datasets bundled with shoez.

Each name below is a dataset, loaded on access as a polars ``DataFrame``::

    from shoez.data import credit

    credit.head()

The files ship inside the package, so a cell copied out of the documentation
runs anywhere shoez is installed. Every access reads the parquet file afresh;
bind the result to a name rather than reaching for the attribute repeatedly.
pandas users can call ``.to_pandas()`` on the result.

The four R-exported datasets (``ames``, ``beans``, ``concrete``, ``credit``)
were written from R with ``arrow::write_parquet`` in the shoez research repo.
``ames_mini`` and ``hotels`` are derived here by the private ``_regenerate_*``
functions at the bottom of this module.
"""

from __future__ import annotations

import io
from importlib.resources import files
from pathlib import Path

import polars as pl

__all__ = ["ames", "ames_mini", "beans", "concrete", "credit"]

_DATA_DIR = Path(__file__).resolve().parent

ames: pl.DataFrame
"""Ames, Iowa housing sales: 2,930 sales x 74 columns.

Target: ``Sale_Price``. Every property sold in Ames between 2006 and 2010,
with 73 columns describing the lot, the building, the sale, and the
location (``Latitude``, ``Longitude``). Mixed numeric and string columns, no
missing values, many near-zero-variance and rare-level columns; the
workhorse regression example in Tidy Modeling with R.

Source: R ``modeldata::ames`` (De Cock, 2011),
https://modeldata.tidymodels.org/reference/ames.html
"""

ames_mini: pl.DataFrame
"""A teaching subset of ``ames``: 2,930 sales x 11 columns.

Target: ``Sale_Price``. The columns Tidy Modeling with R uses in its Ames
recipes (``Neighborhood``, ``Bldg_Type``, ``Gr_Liv_Area``, ``Year_Built``,
``Latitude``, ``Longitude``) plus four chosen to show common preprocessing
problems in one screen: ``Street`` is near-zero variance (12 of 2,930 rows
are ``"Grvl"``), ``Garage_Cars`` and ``Garage_Area`` are highly correlated,
``Neighborhood`` has 7 levels under 1% of rows, and ``Exterior_1st`` has a
level rare enough to land only in a test split. Values are untransformed;
``Sale_Price`` is in dollars.

Source: derived from ``ames``; see ``_regenerate_ames_mini``.
"""

beans: pl.DataFrame
"""Dry beans: 13,611 beans x 17 columns.

Target: ``class`` (7 varieties). 16 numeric shape measurements from images
of each bean, heavily correlated with one another; the dimensionality
reduction example in Tidy Modeling with R, ch. 16.

Source: Koklu & Ozkan (2020) via the R ``beans`` package,
https://github.com/tidymodels/beans
"""

concrete: pl.DataFrame
"""Concrete mixtures: 1,030 mixtures x 9 columns.

Target: ``compressive_strength``. 7 ingredient amounts plus ``age``, all
numeric and complete; the workflow-sets example in Tidy Modeling with R,
ch. 15.

Source: R ``modeldata::concrete`` (Yeh, 1998),
https://modeldata.tidymodels.org/reference/concrete.html
"""

credit: pl.DataFrame
"""Credit applications: 4,454 applications x 14 columns.

Target: ``Status`` (``"good"``/``"bad"``). 9 numeric and 4 string
features with missing values in both (``Income`` has 381 nulls, ``Home``,
``Marital``, and ``Job`` a handful each), so it exercises imputation and
missing-level handling before encoding. Note ``Home`` already has a level
named ``"other"``.

Source: R ``modeldata::credit_data``,
https://modeldata.tidymodels.org/reference/credit_data.html
"""


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> pl.DataFrame:
    if name not in __all__:
        raise AttributeError(f"no dataset named {name!r}")
    raw = files(__name__).joinpath(f"{name}.parquet").read_bytes()
    return pl.read_parquet(io.BytesIO(raw))


# ---------------------------------------------------------------------------
# Regeneration of the derived datasets. Never called at import time; run by
# hand when the source data or the column list changes.

_AMES_MINI_COLUMNS: tuple[str, ...] = (
    "Neighborhood",
    "Bldg_Type",
    "Exterior_1st",
    "Street",
    "Gr_Liv_Area",
    "Year_Built",
    "Garage_Cars",
    "Garage_Area",
    "Latitude",
    "Longitude",
    "Sale_Price",
)


def _regenerate_ames_mini() -> None:  # pyright: ignore[reportUnusedFunction]
    """Rewrite ``ames_mini.parquet`` from the packaged ``ames.parquet``."""
    full = pl.read_parquet(_DATA_DIR / "ames.parquet")
    full.select(_AMES_MINI_COLUMNS).write_parquet(
        _DATA_DIR / "ames_mini.parquet", compression="zstd"
    )
