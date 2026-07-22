"""Canonical end-to-end integration test: the ``design/motivation.md`` housing example.

This is the closing acceptance test for issue #23 (step integration). It wires the
prototype step set into a single real recipe and drives the exact scenario the
motivation document builds its argument around: a housing dataset with mixed column
types where a later step (PCA) operates on columns an earlier step (Dummy) produced,
and a final step (Normalize) operates on columns the PCA step produced.

The pipeline, straight from ``design/motivation.md``:

1. impute the numeric predictors' medians,
2. give categorical predictors an explicit ``"unknown"`` level,
3. one-hot encode the categorical predictors,
4. PCA the dummies **together with just the** ``geo_*`` **numerics** (``sqft`` /
   ``lot_size`` / ``year_built`` flow past untouched), then
5. normalize **everything numeric**, including the fresh PCA components.

The whole point is that steps 4 and 5 name no literal columns: ``has_source`` picks
up whatever the dummy step happened to produce, and the final ``numeric_predictors``
selector is a live query that automatically catches the PCA components that did not
exist when the recipe was declared. Both backends are exercised so the integration is
confirmed backend-agnostic.
"""

from __future__ import annotations

import pandas as pd
import polars as pl
import pytest

from shoez import (
    Recipe,
    Selectors,
    StepDummy,
    StepImputeMedian,
    StepNormalize,
    StepPca,
    StepUnknown,
)

# The numerics that must flow past PCA untouched (they match neither selector).
PLAIN_NUMERIC = ["sqft", "lot_size", "year_built"]
NUM_COMP = 3


def _housing() -> dict[str, list[object]]:
    """A small housing frame with mixed types and a few nulls to impute."""
    return {
        # outcome
        "price": [3.1e5, 4.2e5, 5.0e5, 2.8e5, 6.1e5, 3.9e5, 4.7e5, 5.5e5],
        # plain numerics: imputed then normalized, never PCA'd
        "sqft": [1200.0, 1800.0, None, 950.0, 2400.0, 1600.0, 2000.0, 2200.0],
        "lot_size": [5000.0, 6500.0, 7200.0, None, 8100.0, 5900.0, 6800.0, 7500.0],
        "year_built": [1990.0, 2001.0, 1975.0, 1960.0, 2015.0, None, 1988.0, 2005.0],
        # geo numerics: imputed, then PCA'd together with the dummies
        "geo_lat": [34.05, 34.10, 34.02, None, 34.15, 34.07, 34.11, 34.09],
        "geo_lon": [-118.2, -118.3, -118.1, -118.4, None, -118.25, -118.35, -118.28],
        "geo_elevation": [90.0, 110.0, 85.0, 120.0, 130.0, None, 105.0, 115.0],
        # categoricals: unknown-filled, then one-hot encoded, then PCA'd
        "neighborhood": ["a", "b", "a", "c", None, "b", "c", "a"],
        "style": [
            "ranch",
            "modern",
            "modern",
            None,
            "ranch",
            "modern",
            "ranch",
            "tudor",
        ],
        "roof": ["tile", "shingle", None, "tile", "shingle", "tile", "shingle", "tile"],
    }


def _recipe(df: pd.DataFrame | pl.DataFrame) -> Recipe:
    """The motivation pipeline, declared exactly as the document describes it.

    The document writes ``all_categorical_predictors()`` for the categorical
    columns; the raw housing frame stores them as strings, so the faithful
    selector is ``nominal_predictors()`` (string + categorical + enum). The
    ``categorical_predictors()`` variant matches only true Categorical/Enum
    dtypes and would skip the string columns.
    """
    return (
        Recipe(data=df, outcome=["price"])
        .add(StepImputeMedian(Selectors.numeric_predictors()))
        .add(StepUnknown(Selectors.nominal_predictors()))
        .add(StepDummy(Selectors.nominal_predictors(), id="dummy"))
        .add(
            StepPca(
                Selectors.has_source(step_type=StepDummy)
                | Selectors.starts_with("geo_"),
                num_comp=NUM_COMP,
            )
        )
        .add(StepNormalize(Selectors.numeric_predictors()))
    )


def _housing_test() -> dict[str, list[object]]:
    """A held-out frame with the same schema and its own nulls / new levels."""
    return {
        "price": [4.0e5, 5.2e5],
        "sqft": [None, 1750.0],
        "lot_size": [6000.0, None],
        "year_built": [1995.0, 2010.0],
        "geo_lat": [34.08, None],
        "geo_lon": [-118.22, -118.31],
        "geo_elevation": [100.0, 108.0],
        "neighborhood": ["a", "d"],  # 'd' unseen at fit time
        "style": [None, "modern"],
        "roof": ["tile", None],
    }


class TestMotivationPipeline:
    @pytest.mark.parametrize("ctor", [pd.DataFrame, pl.DataFrame])
    def test_end_to_end(self, ctor: type[pd.DataFrame] | type[pl.DataFrame]) -> None:
        df_train = ctor(_housing())
        df_test = ctor(_housing_test())

        prepped = _recipe(df_train).fit(df_train)
        x_train = prepped.transform(df_train)
        x_test = prepped.transform(df_test)

        assert isinstance(x_train, ctor)
        assert isinstance(x_test, ctor)

        for out, n_rows in ((x_train, 8), (x_test, 2)):
            cols = list(out.columns)
            # The outcome is carried through untouched.
            assert "price" in cols
            # Plain numerics survive PCA and are still present.
            for name in PLAIN_NUMERIC:
                assert name in cols
            # The categoricals were consumed by Dummy; the dummies + geo_* were
            # consumed by PCA. None of them survive to the output.
            for gone in (
                "neighborhood",
                "style",
                "roof",
                "geo_lat",
                "geo_lon",
                "geo_elevation",
            ):
                assert gone not in cols
            # The PCA components exist and were produced by the recipe.
            pcs = [c for c in cols if c.startswith("PC")]
            assert len(pcs) == NUM_COMP
            # Row count is preserved end-to-end.
            assert len(out) == n_rows

    def test_final_numeric_predictors_catches_pca_components(self) -> None:
        # The last StepNormalize is declared with a live ``numeric_predictors``
        # selector; at fit time it must resolve to the plain numerics *and* the
        # PCA components that did not exist when the recipe was declared.
        df = pd.DataFrame(_housing())
        prepped = _recipe(df).fit(df)
        normalize = prepped.steps[-1]
        captured = set(normalize.schema_in().columns)
        assert set(PLAIN_NUMERIC) <= captured
        assert {f"PC{i + 1}" for i in range(NUM_COMP)} <= captured
        # price is an outcome, so a *_predictors selector must never catch it.
        assert "price" not in captured

    def test_transform_agrees_across_backends(self) -> None:
        # pandas and polars must agree column-for-column on the fitted output.
        df_pd = pd.DataFrame(_housing())
        df_pl = pl.DataFrame(_housing())
        out_pd = _recipe(df_pd).fit(df_pd).transform(df_pd)
        out_pl = _recipe(df_pl).fit(df_pl).transform(df_pl)
        assert list(out_pd.columns) == out_pl.columns
        for col in out_pd.columns:
            pd_vals = out_pd[col].tolist()
            pl_vals = out_pl[col].to_list()
            assert pd_vals == pytest.approx(pl_vals), col
