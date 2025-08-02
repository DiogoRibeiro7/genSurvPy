import os
from datetime import datetime
from importlib import metadata

# Project information
project = "gen_surv"
copyright = f"{datetime.now().year}, Diogo Ribeiro"
author = "Diogo Ribeiro"
release = metadata.version("gen_surv")
version = release

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

# MyST Parser configuration
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

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary
autosummary_generate = True

# Napoleon settings
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

# Disable fetching remote inventories when network access is unavailable
if os.environ.get("SKIP_INTERSPHINX", "1") == "1":
    intersphinx_mapping = {}

# HTML theme options
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "canonical_url": "https://gensurvpy.readthedocs.io/",
    "analytics_id": "",
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

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Output file base name for HTML help builder
htmlhelp_basename = "gensurvdoc"

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
