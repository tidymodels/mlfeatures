"""The bundled example datasets in ``shoez.data``.

Each dataset is a lazily loaded polars frame served by the module's
``__getattr__``; the tests pin the shapes, the polars dtype of the one date
column, the derived-column list, the error on an unknown name, and that every
listed dataset carries an attribute docstring (parsed from source with ``ast``,
because an annotation-only name has no runtime object to hold one).
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

import shoez.data

DATA_INIT = Path(shoez.data.__file__)

# (rows, columns) per dataset; the spec's table.
EXPECTED_SHAPES: dict[str, tuple[int, int]] = {
    "ames": (2930, 74),
    "ames_mini": (2930, 11),
    "beans": (13611, 17),
    "concrete": (1030, 9),
    "credit": (4454, 14),
    "hotels": (50000, 23),
}


def test_all_lists_every_expected_dataset() -> None:
    assert set(shoez.data.__all__) == set(EXPECTED_SHAPES)


@pytest.mark.parametrize("name", sorted(EXPECTED_SHAPES))
def test_dataset_loads_as_polars_with_expected_shape(name: str) -> None:
    frame = getattr(shoez.data, name)
    assert isinstance(frame, pl.DataFrame)
    assert frame.shape == EXPECTED_SHAPES[name]


def test_each_access_returns_a_fresh_frame() -> None:
    """No caching: mutating one load must not leak into the next."""
    first = shoez.data.credit
    second = shoez.data.credit
    assert first is not second


def test_unknown_name_raises_attribute_error_naming_it() -> None:
    with pytest.raises(AttributeError, match="no dataset named 'nope'"):
        getattr(shoez.data, "nope")


def test_dir_lists_the_datasets() -> None:
    assert set(dir(shoez.data)) >= set(shoez.data.__all__)


def _attribute_docstrings(source: str) -> dict[str, str]:
    """Map annotated module-level names to the docstring beneath them.

    griffe reads ``name: T`` followed by a string constant as an attribute
    with a docstring; this mirrors that rule so the test checks what the docs
    build will render.
    """
    tree = ast.parse(source)
    found: dict[str, str] = {}
    body = tree.body
    for node, following in zip(body, body[1:]):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        if (
            isinstance(following, ast.Expr)
            and isinstance(following.value, ast.Constant)
            and isinstance(following.value.value, str)
        ):
            found[node.target.id] = following.value.value
    return found


def test_every_dataset_has_an_attribute_docstring() -> None:
    docs = _attribute_docstrings(DATA_INIT.read_text())
    missing = set(shoez.data.__all__) - set(docs)
    assert not missing, f"datasets without a docstring: {sorted(missing)}"
    for name, text in docs.items():
        assert text.strip(), f"empty docstring for {name!r}"


AMES_MINI_COLUMNS = [
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
]


def test_ames_mini_has_the_curated_columns_in_order() -> None:
    assert shoez.data.ames_mini.columns == AMES_MINI_COLUMNS


def test_ames_mini_is_an_untransformed_projection_of_ames() -> None:
    projected = shoez.data.ames.select(AMES_MINI_COLUMNS)
    assert projected.equals(shoez.data.ames_mini)


def test_regenerate_ames_mini_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerating into a scratch directory reproduces the shipped file."""
    scratch = tmp_path / "data"
    scratch.mkdir()
    shoez.data.ames.write_parquet(scratch / "ames.parquet")
    monkeypatch.setattr(shoez.data, "_DATA_DIR", scratch)
    shoez.data._regenerate_ames_mini()
    rebuilt = pl.read_parquet(scratch / "ames_mini.parquet")
    assert rebuilt.equals(shoez.data.ames_mini)


def test_hotels_arrival_date_is_a_date_column() -> None:
    hotels = shoez.data.hotels
    assert hotels["arrival_date"].dtype == pl.Date
    assert hotels["children"].unique().sort().to_list() == ["children", "none"]
    assert hotels.null_count().sum_horizontal().item() == 0
