from importlib.metadata import PackageNotFoundError, version

import pytest

try:
    from gen_surv import __version__
except Exception:  # pragma: no cover - dependency missing
    __version__ = None


def test_package_version_matches_metadata():
    """The exported __version__ should match package metadata."""
    if __version__ is None:
        pytest.skip("gen_surv could not be imported")
    try:
        assert __version__ == version("gen_surv")
    except PackageNotFoundError:
        pytest.skip("package is not installed")
