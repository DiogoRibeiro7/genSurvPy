# conf_pages.py — for GitHub Pages deployment

import os
import sys

# Import the main documentation configuration
# Assumes this file is located in docs/gh_conf/
# and main conf.py is in docs/source/
sys.path.insert(0, os.path.abspath("../source"))
from conf import *  # noqa

# Override base URL for correct links and assets on GitHub Pages
html_baseurl = "https://diogoribeiro7.github.io/packages/gensurvpy/"

# Ensure that GitHub Pages serves _static and other underscored folders
html_extra_path = [".nojekyll"]

# Optional: use a different theme for GitHub Pages if desired
# html_theme = "furo"

# Optional: override theme options for GitHub Pages
# html_theme_options = {
#     "navigation_with_keys": True,
# }

# Optional: override output directory (if not handled in sphinx-build command)
# html_output = "build"
