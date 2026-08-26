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


class TestYamlRoundTrip:
    """YAML is a second encoding over the same dict; it round-trips like JSON."""

    def test_transform_matches_after_yaml_roundtrip(self) -> None:
        rec, df = _multi_step()
        restored = FittedRecipe.from_yaml(rec.to_yaml())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))

    def test_yaml_and_json_decode_to_same_dict(self) -> None:
        rec, _ = _multi_step()
        assert FittedRecipe.from_yaml(rec.to_yaml()).to_dict() == rec.to_dict()

    def test_yaml_is_block_style_and_diffable(self) -> None:
        rec, _ = _multi_step()
        text = rec.to_yaml()
        # block style spreads entries across lines (one key/list item per line)
        # so it diffs cleanly, rather than a single flow-style `{...}` blob.
        assert "format_version:" in text
        assert text.count("\n") > 10

    def test_missing_pyyaml_raises_clear_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import builtins

        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", fake_import)
        rec, _ = _multi_step()
        with pytest.raises(ImportError, match="pyyaml"):
            rec.to_yaml()
        with pytest.raises(ImportError, match="pyyaml"):
            FittedRecipe.from_yaml("format_version: 1")


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


def _zero_arg_selector_names() -> list[str]:
    """Every ``Selectors`` constructor that a no-argument call actually builds.

    ``has_source``/``has_ancestor`` have all-default signatures but reject the
    empty call (they need exactly one of 'step'/'step_type'), so they drop out.
    """
    names: list[str] = []
    for name in dir(Selectors):
        if name.startswith("_"):
            continue
        factory = getattr(Selectors, name)
        if not callable(factory):
            continue
        try:
            factory()
        except (TypeError, ValueError):
            continue
        names.append(name)
    return names


class TestEverySelectorRoundTrips:
    """Every zero-argument ``Selectors`` constructor survives a JSON round-trip.

    The dtype-category selectors (``numeric()``, ``integer()``, ``double()``) once
    encoded to a name the decoder had no entry for (issue #195); enumerating the
    constructors is what caught it, so the test enumerates them too.
    """

    @pytest.mark.parametrize("name", _zero_arg_selector_names())
    def test_roundtrip(self, name: str) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
        selector = getattr(Selectors, name)()
        rec = Recipe().add(StepNormalize(selector)).fit(df)
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


class TestNdarrayNonFiniteRoundTrip:
    """A float ndarray holding NaN / +-inf serializes to valid JSON and round-trips."""

    def test_encode_decode_preserves_non_finite(self) -> None:
        arr = np.array([[1.0, np.nan], [np.inf, -np.inf]])
        text = json.dumps(serialization._encode(arr), allow_nan=False)  # valid JSON
        restored = serialization._decode(json.loads(text))
        np.testing.assert_array_equal(restored, arr)
        assert restored.dtype == arr.dtype


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

    def test_missing_field_falls_back_to_class_default(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        rec = Recipe().add(StepNormalize(["a"])).fit(df)
        payload = rec.to_dict()
        # Simulate an older payload that predates a field carrying a class-level
        # default. ``_empty_selection`` is such a field (``Step._empty_selection
        # = False``); a genuinely optional new field is expected to declare one
        # the same way, which is what lets ``from_state`` recover it.
        del payload["steps"][0]["state"]["_empty_selection"]
        restored = FittedRecipe.from_dict(payload)
        # The dropped field reads back as its class default, not AttributeError.
        assert restored.steps[0]._empty_selection is False  # type: ignore[attr-defined]
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
