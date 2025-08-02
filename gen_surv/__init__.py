"""Top-level package for ``gen_surv``.

This module exposes the main functions and provides access to the package version.
"""

from importlib.metadata import PackageNotFoundError, version

from .aft import gen_aft_log_logistic, gen_aft_log_normal, gen_aft_weibull

# Helper functions
from .bivariate import sample_bivariate_distribution
from .censoring import (
    CensoringModel,
    rexpocens,
    runifcens,
    rweibcens,
    rlognormcens,
    rgammacens,
    WeibullCensoring,
    LogNormalCensoring,
    GammaCensoring,
)
from .cmm import gen_cmm
from .competing_risks import gen_competing_risks, gen_competing_risks_weibull

# Individual generators
from .cphm import gen_cphm
from .export import export_dataset
from .integration import to_sksurv

# Main interface
from .interface import generate
from .mixture import cure_fraction_estimate, gen_mixture_cure
from .piecewise import gen_piecewise_exponential
from .sklearn_adapter import GenSurvDataGenerator
from .tdcm import gen_tdcm
from .thmm import gen_thmm

# Visualization tools (requires matplotlib and lifelines)
try:
    from .visualization import (
        describe_survival,
        plot_covariate_effect,
        plot_hazard_comparison,
        plot_survival_curve,
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
    "gen_aft_log_logistic",
    "gen_competing_risks",
    "gen_competing_risks_weibull",
    "gen_mixture_cure",
    "cure_fraction_estimate",
    "gen_piecewise_exponential",
    # Helpers
    "sample_bivariate_distribution",
    "runifcens",
    "rexpocens",
    "rweibcens",
    "rlognormcens",
    "rgammacens",
    "WeibullCensoring",
    "LogNormalCensoring",
    "GammaCensoring",
    "CensoringModel",
    "export_dataset",
    "to_sksurv",
    "GenSurvDataGenerator",
]

# Add visualization tools to __all__ if available
if _has_visualization:
    __all__.extend(
        [
            "plot_survival_curve",
            "plot_hazard_comparison",
            "plot_covariate_effect",
            "describe_survival",
        ]
    )
