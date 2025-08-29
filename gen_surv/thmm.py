from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validation import validate_gen_thmm_inputs


class TransitionTimes(TypedDict):
    c: float
    t12: float
    t13: float
    t23: float


def calculate_transitions(
    z1: float,
    cens_par: float,
    beta: Sequence[float],
    rate: Sequence[float],
    rfunc: CensoringFunc,
) -> TransitionTimes:
    """
    Calculate transition and censoring times for THMM.

    Parameters:
    - z1 (float): Covariate value.
    - cens_par (float): Censoring parameter.
    - beta (list of float): Coefficients for rate modification (length 3).
    - rate (list of float): Base rates (length 3).
    - rfunc (callable): Censoring function, e.g. runifcens or rexpocens.

    Returns:
    - dict with keys 'c', 't12', 't13', 't23'
    """
    c = rfunc(1, cens_par)[0]
    rate12 = rate[0] * np.exp(beta[0] * z1)
    rate13 = rate[1] * np.exp(beta[1] * z1)
    rate23 = rate[2] * np.exp(beta[2] * z1)

    t12 = np.random.exponential(scale=1 / rate12)
    t13 = np.random.exponential(scale=1 / rate13)
    t23 = np.random.exponential(scale=1 / rate23)

    return {"c": c, "t12": t12, "t13": t13, "t23": t23}


def gen_thmm(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
) -> pd.DataFrame:
    """Generate THMM (Time-Homogeneous Markov Model) survival data.

    Parameters
    ----------
    n : int
        Number of individuals.
    model_cens : {"uniform", "exponential"}
        Censoring model.
    cens_par : float
        Censoring parameter.
    beta : Sequence[float]
        Length-3 regression coefficients.
    covariate_range : float
        Upper bound for the covariate values.
    rate : Sequence[float]
        Length-3 transition rates.

    Returns
    -------
    pd.DataFrame
        Columns = ``["id", "time", "state", "X0"]``.

    Examples
    --------
    >>> from gen_surv.thmm import gen_thmm
    >>> df = gen_thmm(
    ...     n=3,
    ...     model_cens="uniform",
    ...     cens_par=5.0,
    ...     beta=[0.1, 0.2, 0.3],
    ...     covariate_range=1.0,
    ...     rate=[0.1, 0.1, 0.2],
    ... )
    """
    validate_gen_thmm_inputs(n, model_cens, cens_par, beta, covariate_range, rate)
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens
    records = []

    for k in range(n):
        z1 = np.random.uniform(0, covariate_range)
        trans = calculate_transitions(z1, cens_par, beta, rate, rfunc)
        t12, t13, c = trans["t12"], trans["t13"], trans["c"]

        if min(t12, t13) < c:
            if t12 <= t13:
                time, state = t12, 2
            else:
                time, state = t13, 3
        else:
            time, state = c, 1  # censored

        records.append([k + 1, time, state, z1])

    return pd.DataFrame(records, columns=["id", "time", "state", "X0"])
