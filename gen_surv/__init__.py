"""Top-level package for ``gen_surv``.

This module exposes the main functions and provides access to the package version.
"""

from importlib.metadata import PackageNotFoundError, version

# Main interface
from .interface import generate

# Individual generators
from .cphm import gen_cphm
from .cmm import gen_cmm
from .tdcm import gen_tdcm
from .thmm import gen_thmm
from .aft import gen_aft_log_normal, gen_aft_weibull

# Helper functions
from .bivariate import sample_bivariate_distribution
from .censoring import runifcens, rexpocens

# Visualization tools (requires matplotlib and lifelines)
try:
    from .visualization import (
        plot_survival_curve,
        plot_hazard_comparison,
        plot_covariate_effect,
        describe_survival
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

try:
    __version__ = version("gen_surv")
except PackageNotFoundError:  # pragma: no cover - fallback when package not installed
    __version__ = "0.0.0"

__all__ = [
    # Main interface
    "generate",
    "__version__",
    
    # Individual generators
    "gen_cphm",
    "gen_cmm",
    "gen_tdcm",
    "gen_thmm",
    "gen_aft_log_normal",
    "gen_aft_weibull",
    
    # Helpers
    "sample_bivariate_distribution",
    "runifcens",
    "rexpocens",
]

# Add visualization tools to __all__ if available
if _has_visualization:
    __all__.extend([
        "plot_survival_curve",
        "plot_hazard_comparison",
        "plot_covariate_effect",
        "describe_survival"
    ])
    