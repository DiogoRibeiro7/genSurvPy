import os
import sys
from datetime import datetime
from importlib import metadata

# Include src directory for autodoc
sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------

project = "gen_surv"
copyright = f"{datetime.now().year}, Diogo Ribeiro"
author = "Diogo Ribeiro"

# Get version from installed package metadata
try:
    release = metadata.version("gen_surv")
except metadata.PackageNotFoundError:
    release = "0.0.0"  # fallback for local builds
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",  # includes .nojekyll
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# MyST Markdown extensions
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# Disable fetching intersphinx inventories on CI (e.g., for offline builds)
if os.environ.get("SKIP_INTERSPHINX", "1") == "1":
    intersphinx_mapping = {}

# -- HTML output configuration ----------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://gensurvpy.readthedocs.io/",
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Static assets
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Add .nojekyll so GitHub Pages serves _static and other underscored folders
# html_extra_path = [".nojekyll"]

# Required for correct link rendering on GitHub Pages under a subpath
html_baseurl = "https://diogoribeiro7.github.io/packages/gensurvpy/"

# Output file base name for HTML help builder
htmlhelp_basename = "gensurvdoc"

# Copy button config for code blocks
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
