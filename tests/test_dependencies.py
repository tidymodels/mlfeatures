"""Tests for the optional-dependency machinery in :mod:`mlfeatures._dependencies`.

These exercise the mechanism in isolation: a friendly import helper, a
``ModuleNotFoundError`` subclass that existing ``except`` clauses still catch,
and a lazy module proxy that is transparent when its backing library is
installed but raises an actionable error (with an install hint) when it is not.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap

import pytest

from mlfeatures import _dependencies
from mlfeatures._dependencies import (
    MissingDependencyError,
    _LazyModule,
    import_optional,
    module_available,
    require_packages,
)


class TestMissingDependencyError:
    def test_is_module_not_found_error(self) -> None:
        # Existing ``except ModuleNotFoundError`` blocks (e.g. the sklearn guard
        # in compat) must keep catching our richer error.
        assert issubclass(MissingDependencyError, ModuleNotFoundError)


class TestImportOptional:
    def test_returns_real_module_when_present(self) -> None:
        mod = import_optional("json")
        assert mod is importlib.import_module("json")

    def test_raises_for_absent_module(self) -> None:
        with pytest.raises(MissingDependencyError):
            import_optional("mlfeatures_no_such_dependency_zzz")

    def test_error_names_the_module_and_install_hint(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            import_optional("mlfeatures_no_such_dependency_zzz")
        message = str(excinfo.value)
        assert "mlfeatures_no_such_dependency_zzz" in message
        assert "pip install" in message

    def test_known_extra_is_mentioned_in_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force numpy to look uninstalled so we can observe the extra-aware hint
        # without actually uninstalling it from the dev environment.
        real_import = importlib.import_module

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name.split(".")[0] == "numpy":
                raise ModuleNotFoundError("No module named 'numpy'")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_dependencies.importlib, "import_module", fake_import)
        with pytest.raises(MissingDependencyError) as excinfo:
            import_optional("numpy")
        assert "mlfeatures[numpy]" in str(excinfo.value)

    def test_purpose_appears_in_message(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            import_optional(
                "mlfeatures_no_such_dependency_zzz", purpose="to expand polynomials"
            )
        assert "to expand polynomials" in str(excinfo.value)


class TestModuleAvailable:
    def test_true_for_stdlib(self) -> None:
        assert module_available("json") is True

    def test_false_for_absent(self) -> None:
        assert module_available("mlfeatures_no_such_dependency_zzz") is False


class TestLazyModuleProxy:
    def test_numpy_proxy_is_transparent_when_installed(self) -> None:
        # numpy is installed in the dev/CI environment, so the proxy should
        # behave exactly like the real module.
        np = _dependencies.numpy
        arr = np.asarray([1, 2, 3])
        assert list(arr) == [1, 2, 3]
        assert np.ndarray is importlib.import_module("numpy").ndarray

    def test_attribute_access_on_missing_module_raises(self) -> None:
        proxy = _LazyModule("mlfeatures_no_such_dependency_zzz")
        with pytest.raises(MissingDependencyError):
            proxy.some_function

    def test_dunder_access_raises_attribute_error_not_dependency_error(self) -> None:
        # Tooling/inspection probes dunders; those must fail as AttributeError so
        # they don't surface as a misleading "install this" message.
        proxy = _LazyModule("mlfeatures_no_such_dependency_zzz")
        with pytest.raises(AttributeError):
            proxy.__wrapped__

    def test_proxy_caches_attributes_after_first_access(self) -> None:
        # After the real module loads, its attributes are copied onto the proxy so
        # subsequent access resolves via normal __dict__ lookup without re-entering
        # __getattr__ (a hot path: steps touch `np.<...>` repeatedly). Use "json" so
        # the assertion holds regardless of which optional libs are installed.
        proxy = _LazyModule("json")
        assert "dumps" not in vars(proxy)  # not yet imported
        assert proxy.dumps is importlib.import_module("json").dumps
        assert "dumps" in vars(proxy)  # now memoized on the proxy itself


class TestRequirePackages:
    def test_no_error_when_all_present(self) -> None:
        require_packages(["json", "numpy>=1.24"], purpose="by step 'x'")

    def test_raises_naming_package_purpose_and_hint(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(
                ["mlfeatures_no_such_dependency_zzz"], purpose="by step 'x' (step_x)"
            )
        message = str(excinfo.value)
        assert "mlfeatures_no_such_dependency_zzz" in message
        assert "by step 'x' (step_x)" in message
        assert "pip install mlfeatures_no_such_dependency_zzz" in message

    def test_lists_every_missing_package(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(
                ["mlfeatures_no_such_a_zzz", "numpy", "mlfeatures_no_such_b_zzz"],
                purpose="by step 'x'",
            )
        message = str(excinfo.value)
        assert "mlfeatures_no_such_a_zzz" in message
        assert "mlfeatures_no_such_b_zzz" in message
        assert "numpy" not in message  # installed; not reported
        # The two install commands read as a list, not run together.
        assert (
            "`pip install mlfeatures_no_such_a_zzz` and `pip install mlfeatures_no_such_b_zzz`"
            in message
        )

    def test_version_constraint_is_kept_in_the_hint(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(["mlfeatures_no_such_dependency_zzz>=1.3"], purpose="by step")
        assert "pip install 'mlfeatures_no_such_dependency_zzz>=1.3'" in str(excinfo.value)

    def test_known_extra_is_used_for_a_bare_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _dependencies, "module_available", lambda name: name != "numpy"
        )
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(["numpy"], purpose="by step 'x'")
        assert "mlfeatures[numpy]" in str(excinfo.value)

    def test_distribution_name_maps_to_its_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``scikit-learn`` is the requirement name; ``sklearn`` is what imports.
        seen: list[str] = []

        def fake_available(name: str) -> bool:
            seen.append(name)
            return True

        monkeypatch.setattr(_dependencies, "module_available", fake_available)
        require_packages(["scikit-learn>=1.3"], purpose="by step 'x'")
        assert seen == ["sklearn"]

    def test_hint_installs_the_distribution_not_the_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``pip install sklearn`` is a deprecated stub, so the hint must name
        # the distribution even though ``sklearn`` is what gets imported.
        monkeypatch.setattr(_dependencies, "module_available", lambda name: False)
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(["scikit-learn"], purpose="by step 'x'")
        message = str(excinfo.value)
        assert "mlfeatures[sklearn]" in message  # extra is keyed by module name
        assert "pip install scikit-learn" in message
        assert "pip install sklearn" not in message

    def test_known_extra_survives_a_version_constraint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            _dependencies, "module_available", lambda name: name != "numpy"
        )
        with pytest.raises(MissingDependencyError) as excinfo:
            require_packages(["numpy>=1.24"], purpose="by step 'x'")
        message = str(excinfo.value)
        assert "mlfeatures[numpy]" in message
        assert "pip install 'numpy>=1.24'" in message

    def test_distribution_name_is_canonicalized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``-``, ``_``, and ``.`` are interchangeable in a distribution name
        # (PEP 503), so every spelling must find the same module.
        seen: list[str] = []

        def fake_available(name: str) -> bool:
            seen.append(name)
            return True

        monkeypatch.setattr(_dependencies, "module_available", fake_available)
        require_packages(["Scikit_Learn>=1.3"], purpose="by step 'x'")
        assert seen == ["sklearn"]

    def test_dashes_become_underscores_for_unknown_names(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[str] = []

        def fake_available(name: str) -> bool:
            seen.append(name)
            return True

        monkeypatch.setattr(_dependencies, "module_available", fake_available)
        require_packages(["some-plugin"], purpose="by step 'x'")
        assert seen == ["some_plugin"]


# Block ``polars`` (and submodules) from importing, then import the opt-in
# accessor. Run in a subprocess so the import machinery is pristine and polars
# is genuinely unavailable for the duration of the import.
_BLOCK_POLARS_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    class _BlockPolars(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name == "polars" or name.startswith("polars."):
                raise ModuleNotFoundError(f"No module named {name!r}")
            return None

    sys.meta_path.insert(0, _BlockPolars())
    for mod in [m for m in sys.modules if m == "polars" or m.startswith("polars.")]:
        del sys.modules[mod]

    try:
        import mlfeatures.polars  # noqa: F401
    except ModuleNotFoundError as exc:
        assert "polars" in str(exc), str(exc)
        assert "pip install" in str(exc), str(exc)
        print("FRIENDLY_ERROR_OK")
    else:
        raise SystemExit("expected ModuleNotFoundError when polars is absent")
    """
)


class TestOptionalBackendImports:
    """The sanctioned runtime backend imports route through ``import_optional``."""

    def test_polars_accessor_without_polars_raises_friendly_error(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", _BLOCK_POLARS_SCRIPT],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "FRIENDLY_ERROR_OK" in result.stdout
