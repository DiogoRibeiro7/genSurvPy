"""Top-level package for ``gen_surv``.

This module exposes the :func:`generate` function and provides access to the
package version via ``__version__``.
"""

from importlib.metadata import PackageNotFoundError, version

from .interface import generate

try:
    __version__ = version("gen_surv")
except PackageNotFoundError:  # pragma: no cover - fallback when package not installed
    __version__ = "0.0.0"

__all__ = ["generate", "__version__"]

