"""Sphinx configuration for HTML served from a GitHub Pages site.

Reuses ``docs/source/conf.py`` and overrides only what differs when the docs are
published on GitHub Pages instead of Read the Docs. Build with::

    sphinx-build -b html -c docs/gh_conf docs/source docs/build
"""

import os
import sys
from pathlib import Path

_SOURCE_DIR = Path(__file__).resolve().parent.parent / "source"

# Import the main documentation configuration.
sys.path.insert(0, str(_SOURCE_DIR))
from conf import *  # noqa

# Paths in the imported configuration are relative to its own directory, but
# Sphinx resolves them against this file instead, so make them absolute.
html_static_path = [str(_SOURCE_DIR / "_static")]

# ``actions/configure-pages`` exports the site URL as ``DOCS_BASEURL``. The
# fallback keeps the older blog deployment (``sphinx-docs.yaml``) working.
_base_url = (
    os.environ.get("DOCS_BASEURL")
    or "https://diogoribeiro7.github.io/packages/gensurvpy"
)
html_baseurl = _base_url.rstrip("/") + "/"

# Canonical links should point at the site being built, not at Read the Docs.
html_theme_options["canonical_url"] = html_baseurl  # noqa: F405

# ``sphinx.ext.githubpages`` writes the .nojekyll file that makes GitHub Pages
# serve the underscored ``_static`` and ``_sources`` directories.
