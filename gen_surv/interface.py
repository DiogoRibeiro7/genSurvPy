"""
Interface module to unify access to all survival data generators.

Example:
    >>> from gen_surv import generate
    >>> df = generate(model="cphm", n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)
"""

from gen_surv.cphm import gen_cphm
from gen_surv.cmm import gen_cmm
from gen_surv.tdcm import gen_tdcm
from gen_surv.thmm import gen_thmm
from gen_surv.aft import gen_aft_log_normal


_model_map = {
    "cphm": gen_cphm,
    "cmm": gen_cmm,
    "tdcm": gen_tdcm,
    "thmm": gen_thmm,
    "aft_ln": gen_aft_log_normal,
}


def generate(model: str, **kwargs):
    """
    Generic interface to generate survival data from various models.

    Parameters:
        model (str): One of ["cphm", "cmm", "tdcm", "thmm"]
        **kwargs: Arguments forwarded to the selected model generator.

    Returns:
        pd.DataFrame: Simulated survival data.
    """
    model = model.lower()
    if model not in _model_map:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_model_map.keys())}.")
    
    return _model_map[model](**kwargs)
