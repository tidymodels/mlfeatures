"""Smoke tests verifying the package is importable and reports a version."""

import shoez


def test_import() -> None:
    assert shoez is not None


def test_version_is_string() -> None:
    assert isinstance(shoez.__version__, str)
    assert shoez.__version__
