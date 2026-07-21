"""Cross-cutting serialization tests (``design/serialization.md``, issue #125).

The per-step JSON round-trip is enforced uniformly by the ``TestJsonRoundTrip``
class in every ``test_step_*_infrastructure.py``; this file covers the recipe-
level guarantees: the no-closure property, native-selector rejection, NaN
handling, the ``minimize()`` strip mechanism, back-compat field defaults, and a
full multi-step recipe round-trip.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pandas.testing as pt
import polars.selectors as pcs
import pytest
from typing_extensions import Self

from shoez import (
    FittedRecipe,
    Recipe,
    Selectors,
    StepDummy,
    StepImputeMean,
    StepNormalize,
    StepPca,
    serialization,
)
from shoez.step import Step
from shoez.types import AcceptedDtypes, ColSchema, Frame, SchemaInfo


def _multi_step() -> tuple[FittedRecipe, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [4.0, 5.0, 6.0, 7.0],
            "city": ["NYC", "LA", "NYC", "SF"],
        }
    )
    rec = (
        Recipe()
        .add(StepNormalize(Selectors.starts_with("x")))
        .add(StepDummy(["city"]))
        .add(StepPca(["x1", "x2"], num_comp=1))
        .fit(df)
    )
    return rec, df


class TestFullRecipeRoundTrip:
    def test_transform_matches_after_json_roundtrip(self) -> None:
        rec, df = _multi_step()
        restored = FittedRecipe.from_json(rec.to_json())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))

    def test_summary_survives_roundtrip(self) -> None:
        rec, _ = _multi_step()
        restored = FittedRecipe.from_json(rec.to_json())
        # summary() returns a native polars frame; compare with its own equality.
        assert restored.summary().equals(rec.summary())

    def test_dict_roundtrip(self) -> None:
        rec, df = _multi_step()
        restored = FittedRecipe.from_dict(rec.to_dict())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))

    def test_payload_carries_version_metadata(self) -> None:
        rec, _ = _multi_step()
        payload = rec.to_dict()
        assert payload["format_version"] == serialization.FORMAT_VERSION
        assert isinstance(payload["shoez_version"], str)


class TestNoClosures:
    """A fitted recipe serializes to JSON with no closures/environments captured.

    A closure cannot be JSON-encoded, so a payload that both serializes *and*
    contains no function marker proves steps and selectors captured only plain
    data -- the guarantee ``design/serialization.md`` requires.
    """

    def test_json_has_no_function_markers(self) -> None:
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        rec = Recipe().add(StepNormalize(Selectors.starts_with("x"))).fit(df)
        text = rec.to_json()
        json.loads(text)  # valid JSON
        assert "function" not in text
        assert "lambda" not in text
        assert "<" not in text  # no repr'd objects like <function ...> / <object ...>

    def test_name_selector_roundtrips_without_closure(self) -> None:
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
        rec = Recipe().add(StepNormalize(Selectors.starts_with("x"))).fit(df)
        restored = FittedRecipe.from_json(rec.to_json())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))


class TestForeignSelectorRejected:
    """A raw polars/narwhals selector cannot be serialized; the error is clear."""

    def test_native_selector_raises(self) -> None:
        df = pd.DataFrame({"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]})
        rec = Recipe().add(StepNormalize(Selectors.wrap(pcs.numeric()))).fit(df)
        with pytest.raises(TypeError, match="native polars/narwhals selector"):
            rec.to_json()


class TestNaNRoundTrip:
    """Non-finite learned statistics (NaN for an all-null column) round-trip."""

    def test_nan_estimate_survives(self) -> None:
        df = pd.DataFrame({"a": [None, None, None]}, dtype="float64")
        rec = Recipe().add(StepImputeMean(["a"])).fit(df)
        text = rec.to_json()
        assert "__float__" in text  # NaN was tagged, not emitted as a raw token
        restored = FittedRecipe.from_json(text)
        restored_est = restored.steps[0]._estimates  # type: ignore[attr-defined]
        assert np.isnan(restored_est["a"])


class _HeavyStep(Step):
    """A throwaway step that retains a fit-only artifact and drops it in minimize()."""

    def __init__(self) -> None:
        self.columns = ["a"]
        self.id = "heavy"
        self.trained = False
        self.keep_original_cols = False
        self.output_role = None
        self._artifact: list[float] | None = None
        self._schema_in: ColSchema | None = None

    def fit(self, training_data: Frame, info: SchemaInfo) -> "Self":
        # A large training artifact that transform() does not need.
        self._artifact = list(range(1000))
        self._schema_in = ColSchema(columns=["a"], dtypes={"a": "Float64"})
        return self

    def transform(self, new_data: Frame) -> Frame:
        return new_data

    def schema_in(self) -> ColSchema:
        assert self._schema_in is not None
        return self._schema_in

    def schema_out(self) -> ColSchema:
        assert self._schema_in is not None
        return self._schema_in

    def accepts_dtypes(self) -> AcceptedDtypes:
        from shoez.types import NUMERIC_DTYPES

        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        return False

    def minimize(self) -> "Self":
        stripped = _HeavyStep.from_state(self.to_state())
        stripped._artifact = None
        return stripped  # type: ignore[return-value]


class TestMinimizeStrips:
    """minimize() drops fit-only artifacts while preserving transform output."""

    def test_minimize_shrinks_payload_and_preserves_transform(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        rec = Recipe().add(_HeavyStep()).fit(df)
        minimized = rec.minimize()
        pt.assert_frame_equal(minimized.transform(df), rec.transform(df))
        assert len(minimized.to_json()) < len(rec.to_json())
        assert minimized.steps[0]._artifact is None  # type: ignore[attr-defined]

    def test_default_minimize_is_noop(self) -> None:
        rec, df = _multi_step()
        pt.assert_frame_equal(rec.minimize().transform(df), rec.transform(df))


class TestBackCompatDefaults:
    """A payload missing a newer field loads without error (class default kept)."""

    def test_missing_field_uses_new_instance_state(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        rec = Recipe().add(StepNormalize(["a"])).fit(df)
        payload = rec.to_dict()
        # Simulate an older payload that predates the ``output_role`` field.
        del payload["steps"][0]["state"]["output_role"]
        restored = FittedRecipe.from_dict(payload)
        # Reconstruction still succeeds and transforms identically.
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))


class TestModuleAndMethodParity:
    """The module functions and the FittedRecipe methods agree."""

    def test_to_dict_parity(self) -> None:
        rec, _ = _multi_step()
        assert serialization.to_dict(rec) == rec.to_dict()

    def test_ndarray_roundtrip_is_exact(self) -> None:
        rec, _ = _multi_step()
        restored = FittedRecipe.from_json(rec.to_json())
        pca_before = rec.steps[2]._estimates  # type: ignore[attr-defined]
        pca_after = restored.steps[2]._estimates  # type: ignore[attr-defined]
        np.testing.assert_array_equal(pca_after.loadings, pca_before.loadings)
        np.testing.assert_array_equal(pca_after.mean, pca_before.mean)
