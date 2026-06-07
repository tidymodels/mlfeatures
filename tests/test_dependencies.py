"""Tests for the optional-dependency machinery in :mod:`shoez._dependencies`.

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

from shoez import _dependencies
from shoez._dependencies import (
    MissingDependencyError,
    _LazyModule,
    import_optional,
    module_available,
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
            import_optional("shoez_no_such_dependency_zzz")

    def test_error_names_the_module_and_install_hint(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            import_optional("shoez_no_such_dependency_zzz")
        message = str(excinfo.value)
        assert "shoez_no_such_dependency_zzz" in message
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
        assert "shoez[numpy]" in str(excinfo.value)

    def test_purpose_appears_in_message(self) -> None:
        with pytest.raises(MissingDependencyError) as excinfo:
            import_optional(
                "shoez_no_such_dependency_zzz", purpose="to expand polynomials"
            )
        assert "to expand polynomials" in str(excinfo.value)


class TestModuleAvailable:
    def test_true_for_stdlib(self) -> None:
        assert module_available("json") is True

    def test_false_for_absent(self) -> None:
        assert module_available("shoez_no_such_dependency_zzz") is False


class TestLazyModuleProxy:
    def test_numpy_proxy_is_transparent_when_installed(self) -> None:
        # numpy is installed in the dev/CI environment, so the proxy should
        # behave exactly like the real module.
        np = _dependencies.numpy
        arr = np.asarray([1, 2, 3])
        assert list(arr) == [1, 2, 3]
        assert np.ndarray is importlib.import_module("numpy").ndarray

    def test_attribute_access_on_missing_module_raises(self) -> None:
        proxy = _LazyModule("shoez_no_such_dependency_zzz")
        with pytest.raises(MissingDependencyError):
            proxy.some_function

    def test_dunder_access_raises_attribute_error_not_dependency_error(self) -> None:
        # Tooling/inspection probes dunders; those must fail as AttributeError so
        # they don't surface as a misleading "install this" message.
        proxy = _LazyModule("shoez_no_such_dependency_zzz")
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
        import shoez.polars  # noqa: F401
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
