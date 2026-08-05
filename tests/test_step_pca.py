"""Tests for fit_pca / apply_pca (the fit + apply halves of the PCA step)."""

from __future__ import annotations

from typing import Any

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from shoez.step_pca import StepPca, apply_pca, fit_pca


def _df(*cols: tuple[str, list[Any]]) -> nw.DataFrame[Any]:
    return nw.from_native(pd.DataFrame(dict(cols)))


def _sample() -> nw.DataFrame[Any]:
    return _df(("x", [1.0, 2.0, 3.0, 4.0]), ("y", [2.1, 3.9, 6.1, 8.0]))


class TestFitPca:
    def test_loadings_shape(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        assert fit.loadings.shape == (2, 2)
        assert fit.num_comp == 2
        assert fit.columns == ["x", "y"]

    def test_num_comp_capped_at_features(self) -> None:
        # Only 2 columns, so requesting 5 components yields at most 2.
        fit = fit_pca(_sample(), num_comp=5)
        assert fit.num_comp == 2

    def test_mean_is_column_means(self) -> None:
        fit = fit_pca(_sample(), num_comp=1)
        assert fit.mean == pytest.approx([2.5, 5.025])

    @pytest.mark.parametrize("bad", [0, -1, 2.0, True])
    def test_invalid_num_comp_raises(self, bad: Any) -> None:
        with pytest.raises(ValueError):
            fit_pca(_sample(), num_comp=bad)


class TestApplyPca:
    def test_replaces_inputs_with_components(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        result = apply_pca(_sample(), fit, prefix="PC")
        assert result.columns == ["PC1", "PC2"]

    def test_components_are_centered(self) -> None:
        # Projections of centered data sum to ~0 per component.
        fit = fit_pca(_sample(), num_comp=2)
        result = apply_pca(_sample(), fit, prefix="PC")
        assert result["PC1"].to_list() == pytest.approx(
            [v - np.mean(result["PC1"].to_list()) for v in result["PC1"].to_list()],
            abs=1e-9,
        )

    def test_other_columns_pass_through(self) -> None:
        df = _df(
            ("id", [10, 20, 30, 40]),
            ("x", [1.0, 2.0, 3.0, 4.0]),
            ("y", [2.1, 3.9, 6.1, 8.0]),
        )
        fit = fit_pca(df.select(["x", "y"]), num_comp=2)
        result = apply_pca(df, fit, prefix="PC")
        assert result.columns == ["id", "PC1", "PC2"]
        assert result["id"].to_list() == [10, 20, 30, 40]

    def test_custom_prefix(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        result = apply_pca(_sample(), fit, prefix="comp")
        assert result.columns == ["comp1", "comp2"]

    def test_keep_original_cols_retains_inputs(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        result = apply_pca(_sample(), fit, prefix="PC", keep_original_cols=True)
        assert set(result.columns) == {"x", "y", "PC1", "PC2"}
        assert result["x"].to_list() == [1.0, 2.0, 3.0, 4.0]

    def test_keep_original_cols_does_not_change_components(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        dropped = apply_pca(_sample(), fit, prefix="PC")
        kept = apply_pca(_sample(), fit, prefix="PC", keep_original_cols=True)
        assert kept["PC1"].to_list() == pytest.approx(dropped["PC1"].to_list())


class TestBackends:
    def test_polars_in_polars_out(self) -> None:
        df = nw.from_native(
            pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [2.1, 3.9, 6.1, 8.0]})
        )
        fit = fit_pca(df, num_comp=2)
        result = apply_pca(df, fit, prefix="PC")
        assert isinstance(result.to_native(), pl.DataFrame)
        assert result.columns == ["PC1", "PC2"]

    def test_pandas_in_pandas_out(self) -> None:
        fit = fit_pca(_sample(), num_comp=2)
        result = apply_pca(_sample(), fit, prefix="PC")
        assert isinstance(result.to_native(), pd.DataFrame)


class TestConstructor:
    def test_default_id(self) -> None:
        assert StepPca(["x", "y"]).id == "pca"

    def test_default_num_comp(self) -> None:
        assert StepPca(["x", "y"]).num_comp == 5

    def test_required_pkgs(self) -> None:
        assert StepPca(["x", "y"]).required_pkgs() == ["numpy"]

    def test_default_output_role_is_predictor(self) -> None:
        assert StepPca(["x", "y"]).output_role == "predictor"

    def test_keep_original_cols_defaults_to_false(self) -> None:
        assert StepPca(["x", "y"]).keep_original_cols is False

    def test_keep_original_cols_is_settable(self) -> None:
        assert StepPca(["x", "y"], keep_original_cols=True).keep_original_cols is True
