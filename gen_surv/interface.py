"""
Interface module to unify access to all survival data generators.

Example:
    >>> from gen_surv import generate
    >>> df = generate(model="cphm", n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
"""

from typing import Any, Literal, Protocol, Dict

import pandas as pd

from gen_surv.aft import gen_aft_log_logistic, gen_aft_log_normal, gen_aft_weibull
from gen_surv.cmm import gen_cmm
from gen_surv.competing_risks import gen_competing_risks, gen_competing_risks_weibull
from gen_surv.cphm import gen_cphm
from gen_surv.mixture import gen_mixture_cure
from gen_surv.piecewise import gen_piecewise_exponential
from gen_surv.tdcm import gen_tdcm
from gen_surv.thmm import gen_thmm
from ._validation import ensure_in_choices

# Type definitions for model names
ModelType = Literal[
    "cphm",
    "cmm",
    "tdcm",
    "thmm",
    "aft_ln",
    "aft_weibull",
    "aft_log_logistic",
    "competing_risks",
    "competing_risks_weibull",
    "mixture_cure",
    "piecewise_exponential",
]

# Interface for generator callables
class DataGenerator(Protocol):
    def __call__(self, **kwargs: Any) -> pd.DataFrame: ...


# Map model names to their generator functions
_model_map: Dict[ModelType, DataGenerator] = {
    "cphm": gen_cphm,
    "cmm": gen_cmm,
    "tdcm": gen_tdcm,
    "thmm": gen_thmm,
    "aft_ln": gen_aft_log_normal,
    "aft_weibull": gen_aft_weibull,
    "aft_log_logistic": gen_aft_log_logistic,
    "competing_risks": gen_competing_risks,
    "competing_risks_weibull": gen_competing_risks_weibull,
    "mixture_cure": gen_mixture_cure,
    "piecewise_exponential": gen_piecewise_exponential,
}


def generate(model: ModelType, **kwargs: Any) -> pd.DataFrame:
    """Generate survival data from a specific model.

    Args:
        model: Name of the generator to run. Must be one of ``cphm``, ``cmm``,
            ``tdcm``, ``thmm``, ``aft_ln``, ``aft_weibull``, ``aft_log_logistic``,
            ``competing_risks``, ``competing_risks_weibull``, ``mixture_cure``,
            or ``piecewise_exponential``.
        **kwargs: Arguments forwarded to the chosen generator. These vary by model.

            - cphm: n, model_cens, cens_par, beta, covariate_range
            - cmm: n, model_cens, cens_par, beta, covariate_range, rate
            - tdcm: n, dist, corr, dist_par, model_cens, cens_par, beta, lam
            - thmm: n, model_cens, cens_par, beta, covariate_range, rate
            - aft_ln: n, beta, sigma, model_cens, cens_par, seed
            - aft_weibull: n, beta, shape, scale, model_cens, cens_par, seed
            - aft_log_logistic: n, beta, shape, scale, model_cens, cens_par, seed
            - competing_risks: n, n_risks, baseline_hazards, betas, covariate_dist, etc.
            - competing_risks_weibull: n, n_risks, shape_params, scale_params, betas, etc.
            - mixture_cure: n, cure_fraction, baseline_hazard, betas_survival,
              betas_cure, etc.
            - piecewise_exponential: n, breakpoints, hazard_rates, betas, etc.

    Returns:
        pd.DataFrame: Simulated survival data with columns specific to the chosen model.
            All models include time/duration and status columns.

    Raises:
        ChoiceError: If an unknown model name is provided.
    """
    model_lower = model.lower()
    ensure_in_choices(model_lower, "model", _model_map.keys())
    return _model_map[model_lower](**kwargs)
