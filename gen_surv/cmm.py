from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validate import validate_gen_cmm_inputs


class EventTimes(TypedDict):
    t12: float
    t13: float
    t23: float


def generate_event_times(
    z1: float, beta: Sequence[float], rate: Sequence[float]
) -> EventTimes:
    """
    Generate event times for a continuous-time multi-state Markov model.

    Parameters:
    - z1 (float): Covariate value
    - beta (list of float): List of 3 beta coefficients
    - rate (list of float): List of 6 transition rate parameters

    Returns:
    - dict: {'t12': float, 't13': float, 't23': float}
    """
    u = np.random.uniform()
    t12 = (-np.log(1 - u) / (rate[0] * np.exp(beta[0] * z1))) ** (1 / rate[1])

    u = np.random.uniform()
    t13 = (-np.log(1 - u) / (rate[2] * np.exp(beta[1] * z1))) ** (1 / rate[3])

    u = np.random.uniform()
    t23 = (-np.log(1 - u) / (rate[4] * np.exp(beta[2] * z1))) ** (1 / rate[5])

    return {"t12": t12, "t13": t13, "t23": t23}


def gen_cmm(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
) -> pd.DataFrame:
    """
    Generate survival data using a continuous-time Markov model (CMM).

    Parameters:
    - n (int): Number of individuals.
    - model_cens (str): "uniform" or "exponential".
    - cens_par (float): Parameter for censoring.
    - beta (list): Regression coefficients (length 3).
    - covariate_range (float): Upper bound for the covariate values.
    - rate (list): Transition rates (length 6).

    Returns:
    - pd.DataFrame with columns: id, start, stop, status, X0, transition
    """
    validate_gen_cmm_inputs(n, model_cens, cens_par, beta, covariate_range, rate)

    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens

    z1 = np.random.uniform(0, covariate_range, size=n)
    c = rfunc(n, cens_par)

    u = np.random.uniform(size=(3, n))
    t12 = (-np.log(1 - u[0]) / (rate[0] * np.exp(beta[0] * z1))) ** (1 / rate[1])
    t13 = (-np.log(1 - u[1]) / (rate[2] * np.exp(beta[1] * z1))) ** (1 / rate[3])

    first_event = np.minimum(t12, t13)
    censored = first_event >= c

    status = (~censored).astype(int)
    transition = np.where(censored, np.nan, np.where(t12 <= t13, 1, 2))
    stop = np.where(censored, c, first_event)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "start": np.zeros(n),
            "stop": stop,
            "status": status,
            "X0": z1,
            "transition": transition,
        }
    )
