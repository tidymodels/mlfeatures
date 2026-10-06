"""Smoke tests verifying the package is importable and reports a version."""

import mlfeatures


def test_import() -> None:
    assert mlfeatures is not None


def test_version_is_string() -> None:
    assert isinstance(mlfeatures.__version__, str)
    assert mlfeatures.__version__
