from importlib.metadata import version
from gen_surv import __version__


def test_package_version_matches_metadata():
    """The exported __version__ should match package metadata."""
    assert __version__ == version("gen_surv")
