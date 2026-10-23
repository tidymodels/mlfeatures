"""Cross-cutting serialization tests (``design/serialization.md``, issue #125).

The per-step JSON round-trip is enforced uniformly by the ``TestJsonRoundTrip``
class in every ``test_step_*_infrastructure.py``; this file covers the recipe-
level guarantees: the no-closure property, native-selector rejection, NaN
handling, the ``minimize()`` strip mechanism, back-compat field defaults, and a
full multi-step recipe round-trip.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import types
import warnings
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas.testing as pt
import polars.selectors as pcs
import pytest
from syrupy.assertion import SnapshotAssertion
from typing_extensions import Self

import mlfeatures.recipe as recipe_module
from mlfeatures import (
    FittedRecipe,
    Recipe,
    Selectors,
    StepDummy,
    StepImputeMean,
    StepNormalize,
    StepPca,
    serialization,
)
from mlfeatures.step import Step
from mlfeatures.types import AcceptedDtypes, ColSchema, Frame, Mode, SchemaInfo


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
        assert isinstance(payload["mlfeatures_version"], str)


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


class TestKeepOriginalColsRoundTrip:
    """``keep_original_cols`` survives serialization despite being read-only (#176)."""

    def test_fixed_value_survives_and_stays_read_only(self) -> None:
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        rec = Recipe().add(StepNormalize(["a"])).fit(df)
        restored = FittedRecipe.from_json(rec.to_json())
        step = restored.steps[0]
        assert step.keep_original_cols is False
        with pytest.raises(AttributeError):
            step.keep_original_cols = True  # type: ignore[misc]

    def test_configured_value_survives(self) -> None:
        df = pd.DataFrame({"x": ["a", "b", "a"]})
        rec = Recipe().add(StepDummy(["x"], keep_original_cols=True)).fit(df)
        restored = FittedRecipe.from_json(rec.to_json())
        assert restored.steps[0].keep_original_cols is True

    def test_a_payload_written_before_the_property_still_loads(self) -> None:
        # The attribute used to serialize under its public name; a stored recipe
        # from then must not trip over the read-only property on the way back in.
        step = StepNormalize.from_state({"columns": ["a"], "keep_original_cols": True})
        assert step.keep_original_cols is True


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

    mode = Mode.ONE_TO_ONE

    def __init__(self) -> None:
        self.columns = ["a"]
        super().__init__("heavy", "heavy")
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
        from mlfeatures.types import NUMERIC_DTYPES

        return set(NUMERIC_DTYPES)

    @property
    def produces_sparse(self) -> bool:
        return False

    def minimize(self) -> "Self":
        stripped = _HeavyStep.from_state(self.to_state())
        stripped._artifact = None
        return stripped  # type: ignore[return-value]


@pytest.mark.filterwarnings("ignore:recipe names class")
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


class TestFitStatePlaceholders:
    """The base class's fit-state placeholders ride along in every step's state."""

    PLACEHOLDERS = ("_estimates", "_schema_in", "_schema_out", "_output_parents")

    def test_untrained_step_state_carries_every_placeholder_as_none(self) -> None:
        state = StepNormalize(["a"]).to_state()
        for attr in self.PLACEHOLDERS:
            assert state[attr] is None

    def test_unused_placeholders_stay_none_through_a_round_trip(self) -> None:
        """``StepNormalize`` fills ``_schema_in`` but never ``_schema_out``."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        fitted = Recipe().add(StepNormalize(["a"])).fit(df)
        restored = FittedRecipe.from_dict(fitted.to_dict()).steps[0]
        assert restored._schema_out is None  # pyright: ignore[reportPrivateUsage]
        assert restored._output_parents is None  # pyright: ignore[reportPrivateUsage]
        assert restored._schema_in is not None  # pyright: ignore[reportPrivateUsage]

    def test_a_payload_written_before_the_hoisting_still_loads(self) -> None:
        """Older payloads lack ``_schema_out`` and ``_output_parents`` on such a step.

        ``from_state`` bypasses ``__init__``, so a missing key is simply absent
        rather than ``None``. That is harmless for a step that never reads it:
        the restored step still transforms identically.
        """
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        fitted = Recipe().add(StepNormalize(["a"])).fit(df)
        payload = fitted.to_dict()
        for key in ("_schema_out", "_output_parents"):
            del payload["steps"][0]["state"][key]
        restored = FittedRecipe.from_dict(payload)
        pt.assert_frame_equal(restored.transform(df), fitted.transform(df))


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


def _plugin_entry_point(value: str) -> EntryPoint:
    return EntryPoint(name="heavy", value=value, group="mlfeatures.steps")


def _register(monkeypatch: pytest.MonkeyPatch, *entry_points: object) -> None:
    """Make ``entry_points`` the installed ``mlfeatures.steps`` registrations."""
    monkeypatch.setattr(serialization, "_step_entry_points", lambda: list(entry_points))


@dataclasses.dataclass(frozen=True)
class _Note:
    """Fit state a plugin step keeps as a dataclass."""

    value: int


def _foreign_dataclass(monkeypatch: pytest.MonkeyPatch) -> type:
    """A dataclass defined in a package other than this one (``other_pkg_s1``)."""
    foreign = dataclasses.make_dataclass("Foreign", [("value", int)])
    foreign.__module__ = "other_pkg_s1"
    module = types.ModuleType("other_pkg_s1")
    module.Foreign = foreign  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "other_pkg_s1", module)
    return foreign


def _heavy_fitted(note: object = None) -> FittedRecipe:
    fitted = Recipe().add(_HeavyStep()).fit(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))
    if note is not None:
        fitted.steps[0]._note = note  # type: ignore[attr-defined]
    return fitted


def _heavy_recipe_payload(note: object = None) -> dict[str, Any]:
    # Saving warns that the step is unregistered; that is what these tests load.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _heavy_fitted(note).to_dict()


def _load_error(payload: dict[str, Any], **kwargs: Any) -> str:
    """The message of the ``ValueError`` loading ``payload`` raises."""
    with pytest.raises(ValueError) as excinfo:
        FittedRecipe.from_dict(payload, **kwargs)
    return str(excinfo.value)


def _pca_dataclass_name() -> str:
    """The qualified name of the fit dataclass ``StepPca`` stores."""
    rec, _ = _multi_step()
    return rec.to_dict()["steps"][2]["state"]["_estimates"]["__dataclass__"]


class TestPayloadCannotRunCode:
    """A payload can only name mlfeatures classes and registered plugin steps.

    The class names in a payload (a step's ``type``, a fit ``__dataclass__``, a
    selector's ``step_type``) used to be imported and called as given, so a
    crafted JSON or YAML file ran an arbitrary callable on load.
    """

    @staticmethod
    def _tampered(key: str, value: object) -> dict[str, Any]:
        rec, _ = _multi_step()
        payload = rec.to_dict()
        payload["steps"][0]["state"][key] = value
        return payload

    def test_dataclass_tag_cannot_call_an_arbitrary_callable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        snapshot: SnapshotAssertion,
    ) -> None:
        marker = tmp_path / "pwned"
        attacks = [
            ("builtins.print", {"end": "pwned"}),
            ("subprocess.call", {"args": ["touch", str(marker)]}),
        ]
        messages = [
            _load_error(
                self._tampered(
                    "injected", {"__dataclass__": qualname, "fields": fields}
                )
            )
            for qualname, fields in attacks
        ]
        assert messages == snapshot
        assert not marker.exists()
        assert capsys.readouterr().out == ""

    def test_the_same_attack_through_json(
        self, tmp_path: Path, snapshot: SnapshotAssertion
    ) -> None:
        marker = tmp_path / "pwned"
        payload = self._tampered(
            "injected",
            {
                "__dataclass__": "subprocess.call",
                "fields": {"args": ["touch", str(marker)]},
            },
        )
        with pytest.raises(ValueError) as excinfo:
            FittedRecipe.from_json(json.dumps(payload))
        assert str(excinfo.value) == snapshot
        assert not marker.exists()

    def test_the_same_attack_through_yaml(
        self, tmp_path: Path, snapshot: SnapshotAssertion
    ) -> None:
        yaml = pytest.importorskip("yaml")
        marker = tmp_path / "pwned"
        payload = self._tampered(
            "injected",
            {
                "__dataclass__": "subprocess.call",
                "fields": {"args": ["touch", str(marker)]},
            },
        )
        with pytest.raises(ValueError) as excinfo:
            FittedRecipe.from_yaml(yaml.safe_dump(payload))
        assert str(excinfo.value) == snapshot
        assert not marker.exists()

    def test_step_type_cannot_name_an_arbitrary_callable(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec, _ = _multi_step()
        payload = rec.to_dict()
        payload["steps"][0]["type"] = "builtins.print"
        assert _load_error(payload) == snapshot

    def test_untrusted_module_is_never_imported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        snapshot: SnapshotAssertion,
    ) -> None:
        marker = tmp_path / "imported"
        (tmp_path / "evil_mod_s1.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
            "class Thing:\n    pass\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        payload = self._tampered(
            "injected", {"__dataclass__": "evil_mod_s1.Thing", "fields": {}}
        )
        assert _load_error(payload) == snapshot
        assert not marker.exists()
        assert "evil_mod_s1" not in sys.modules

    def test_step_type_must_be_a_step_subclass(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec, _ = _multi_step()
        payload = rec.to_dict()
        payload["steps"][0]["type"] = "mlfeatures.recipe.Recipe"
        assert _load_error(payload) == snapshot

    def test_dataclass_tag_must_name_a_dataclass(
        self, snapshot: SnapshotAssertion
    ) -> None:
        payload = self._tampered(
            "injected", {"__dataclass__": "mlfeatures.recipe.Recipe", "fields": {}}
        )
        assert _load_error(payload) == snapshot

    def test_a_trusted_module_cannot_launder_a_third_party_class(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
    ) -> None:
        # ``_Note`` is defined in this test module, which is not trusted, but a
        # name under ``mlfeatures`` resolves to it.
        monkeypatch.setattr(recipe_module, "Laundered", _Note, raising=False)
        payload = self._tampered(
            "injected",
            {"__dataclass__": "mlfeatures.recipe.Laundered", "fields": {"value": 1}},
        )
        assert _load_error(payload) == snapshot

    def test_selector_step_type_is_restricted(
        self, snapshot: SnapshotAssertion
    ) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "city": ["a", "b", "a"]})
        rec = (
            Recipe()
            .add(StepDummy(["city"]))
            .add(StepNormalize(Selectors.has_source(step_type=StepDummy)))
            .fit(df)
        )
        text = rec.to_json()
        restored = FittedRecipe.from_json(text)
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))

        marker = '"step_type": "mlfeatures.step_dummy.StepDummy"'
        assert marker in text
        messages = [
            _load_error(json.loads(text.replace(marker, f'"step_type": "{name}"')))
            for name in ("builtins.print", "mlfeatures.recipe.Recipe")
        ]
        assert messages == snapshot

    def test_state_key_cannot_replace_a_method(
        self, snapshot: SnapshotAssertion
    ) -> None:
        messages = [
            _load_error(self._tampered(key, 5)) for key in ("transform", "__class__")
        ]
        assert messages == snapshot

    def test_a_trusted_class_does_not_open_the_load_to_other_callables(self) -> None:
        payload = self._tampered(
            "injected", {"__dataclass__": "builtins.print", "fields": {}}
        )
        with pytest.raises(ValueError):
            FittedRecipe.from_dict(payload, trusted=[_HeavyStep])


class TestMalformedPayloads:
    """A payload that is malformed raises ``ValueError``, as one that names a bad class does."""

    def test_malformed_step_and_dataclass_payloads(
        self, snapshot: SnapshotAssertion
    ) -> None:
        rec, _ = _multi_step()
        name = _pca_dataclass_name()

        def with_step(**changes: object) -> dict[str, Any]:
            payload = rec.to_dict()
            payload["steps"][0].update(changes)
            return payload

        def with_state(value: object) -> dict[str, Any]:
            payload = rec.to_dict()
            payload["steps"][0]["state"]["injected"] = value
            return payload

        payloads = {
            "step type is not a string": with_step(type=123),
            "step type is missing": {
                **rec.to_dict(),
                "steps": [{"state": rec.to_dict()["steps"][0]["state"]}],
            },
            "step state is not a mapping": with_step(state=[1, 2]),
            "step is not a mapping": {**rec.to_dict(), "steps": ["x"]},
            "dataclass name is not a string": with_state(
                {"__dataclass__": 5, "fields": {}}
            ),
            "dataclass fields is not a mapping": with_state(
                {"__dataclass__": name, "fields": 5}
            ),
            "dataclass fields is missing": with_state({"__dataclass__": name}),
            "dataclass field is unknown": with_state(
                {"__dataclass__": name, "fields": {"nope": 1}}
            ),
        }
        assert {label: _load_error(p) for label, p in payloads.items()} == snapshot


class TestRegisteredPlugins:
    """Steps and fit state from a package that registers under ``mlfeatures.steps``."""

    def test_unregistered_plugin_step_is_refused(
        self, snapshot: SnapshotAssertion
    ) -> None:
        assert _load_error(_heavy_recipe_payload()) == snapshot

    def test_registered_plugin_step_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        restored = FittedRecipe.from_dict(_heavy_recipe_payload())
        assert isinstance(restored.steps[0], _HeavyStep)

    def test_registered_plugin_fit_dataclass_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        restored = FittedRecipe.from_dict(_heavy_recipe_payload(_Note(3)))
        assert restored.steps[0]._note == _Note(3)  # type: ignore[attr-defined]

    def test_dataclass_from_another_package_is_refused_unless_trusted(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
    ) -> None:
        foreign = _foreign_dataclass(monkeypatch)
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        payload = _heavy_recipe_payload(foreign(1))
        assert _load_error(payload) == snapshot
        restored = FittedRecipe.from_dict(payload, trusted=[foreign])
        assert restored.steps[0]._note == foreign(1)  # type: ignore[attr-defined]

    def test_step_re_exported_from_another_package_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The entry point names ``plugin_api_s1:_HeavyStep``, but the class is
        # defined (and so serialized) under this test module's name.
        api = types.ModuleType("plugin_api_s1")
        api._HeavyStep = _HeavyStep  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "plugin_api_s1", api)
        _register(monkeypatch, _plugin_entry_point("plugin_api_s1:_HeavyStep"))
        restored = FittedRecipe.from_dict(_heavy_recipe_payload())
        assert isinstance(restored.steps[0], _HeavyStep)

    def test_a_malformed_entry_point_does_not_break_loading(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
    ) -> None:
        _register(
            monkeypatch,
            EntryPoint(name="x", value="not a valid value!!", group="mlfeatures.steps"),
        )
        rec, df = _multi_step()
        restored = FittedRecipe.from_dict(rec.to_dict())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))
        assert _load_error(_heavy_recipe_payload()) == snapshot

    def test_built_in_steps_never_read_the_entry_points(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def unexpected() -> list[object]:
            raise AssertionError("the entry points were read")

        monkeypatch.setattr(serialization, "_step_entry_points", unexpected)
        rec, df = _multi_step()
        restored = FittedRecipe.from_json(rec.to_json())
        pt.assert_frame_equal(restored.transform(df), rec.transform(df))

    def test_a_refused_name_loads_no_plugin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Spy:
            module = "spy_plugin"
            attr = "SpyStep"
            loaded = False

            def load(self) -> object:
                Spy.loaded = True
                raise ImportError("broken plugin")

        _register(monkeypatch, Spy())
        rec, _ = _multi_step()
        payload = rec.to_dict()
        payload["steps"][0]["type"] = "builtins.print"
        with pytest.raises(ValueError):
            FittedRecipe.from_dict(payload)
        assert not Spy.loaded

        # A name a plugin does register is tried, and a plugin that fails to
        # import leaves the usual ``ValueError``, not its own ``ImportError``.
        payload["steps"][0]["type"] = "elsewhere.SpyStep"
        with pytest.raises(ValueError):
            FittedRecipe.from_dict(payload)
        assert Spy.loaded

    def test_default_loading_covers_mlfeatures_and_registered_packages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert serialization._is_loaded_by_default(StepNormalize)  # pyright: ignore[reportPrivateUsage]
        assert not serialization._is_loaded_by_default(_HeavyStep)  # pyright: ignore[reportPrivateUsage]
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        assert serialization._is_loaded_by_default(_HeavyStep)  # pyright: ignore[reportPrivateUsage]


class TestTrustedArgument:
    """``trusted`` loads a custom step that is not registered, for that call only."""

    def test_trusted_loads_an_unregistered_step(self) -> None:
        payload = _heavy_recipe_payload(_Note(3))
        restored = FittedRecipe.from_dict(payload, trusted=[_HeavyStep])
        assert isinstance(restored.steps[0], _HeavyStep)
        assert restored.steps[0]._note == _Note(3)  # type: ignore[attr-defined]

    def test_trusted_applies_to_json(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            text = _heavy_fitted().to_json()
        restored = FittedRecipe.from_json(text, trusted=[_HeavyStep])
        assert isinstance(restored.steps[0], _HeavyStep)
        assert isinstance(
            serialization.from_json(text, trusted=[_HeavyStep]).steps[0], _HeavyStep
        )

    def test_trusted_applies_to_yaml(self) -> None:
        pytest.importorskip("yaml")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            text = _heavy_fitted().to_yaml()
        restored = FittedRecipe.from_yaml(text, trusted=[_HeavyStep])
        assert isinstance(restored.steps[0], _HeavyStep)

    def test_trusted_does_not_outlive_the_call(self) -> None:
        payload = _heavy_recipe_payload()
        FittedRecipe.from_dict(payload, trusted=[_HeavyStep])
        with pytest.raises(ValueError):
            FittedRecipe.from_dict(payload)


class TestInProcessStateIsNotRestricted:
    """``from_state`` on a step's own ``to_state`` names nothing a payload chose."""

    def test_step_with_an_unregistered_fit_dataclass_round_trips(self) -> None:
        step = _heavy_fitted(_Note(3)).steps[0]
        restored = _HeavyStep.from_state(step.to_state())
        assert restored._note == _Note(3)  # type: ignore[attr-defined]

    def test_minimize_keeps_an_unregistered_fit_dataclass(self) -> None:
        minimized = _heavy_fitted(_Note(3)).minimize()
        assert minimized.steps[0]._note == _Note(3)  # type: ignore[attr-defined]

    def test_selector_naming_an_unregistered_step_round_trips(self) -> None:
        step = StepNormalize(Selectors.has_source(step_type=_HeavyStep))
        restored = StepNormalize.from_state(step.to_state())
        assert restored.columns._step_type is _HeavyStep  # type: ignore[attr-defined]


class TestSavingWarns:
    """A recipe that ``from_dict`` would refuse by default says so when it is saved."""

    def test_unregistered_step_warns(self, snapshot: SnapshotAssertion) -> None:
        with pytest.warns(UserWarning) as record:
            _heavy_fitted().to_json()
        assert [str(w.message) for w in record] == snapshot

    def test_unregistered_fit_dataclass_warns(
        self, monkeypatch: pytest.MonkeyPatch, snapshot: SnapshotAssertion
    ) -> None:
        foreign = _foreign_dataclass(monkeypatch)
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        with pytest.warns(UserWarning) as record:
            _heavy_fitted(foreign(1)).to_dict()
        assert [str(w.message) for w in record] == snapshot

    def test_built_in_and_registered_steps_do_not_warn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _register(monkeypatch, _plugin_entry_point(f"{__name__}:_HeavyStep"))
        rec, _ = _multi_step()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rec.to_json()
            _heavy_fitted(_Note(3)).to_json()


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
