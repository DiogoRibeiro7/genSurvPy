"""
Cox Proportional Hazards Model (CPHM) data generation.

This module provides functions to generate survival data following the
Cox Proportional Hazards Model with various censoring mechanisms.
"""

from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd

from gen_surv.censoring import rexpocens, runifcens
from gen_surv.validate import validate_gen_cphm_inputs


def generate_cphm_data(
    n: int,
    rfunc: Callable[[int, float], np.ndarray],
    cens_par: float,
    beta: float,
    covariate_range: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate data from a Cox Proportional Hazards Model (CPHM).

    Parameters
    ----------
    n : int
        Number of samples to generate.
    rfunc : callable
        Function to generate censoring times, must accept (size, cens_par).
    cens_par : float
        Parameter passed to the censoring function.
    beta : float
        Coefficient for the covariate.
    covariate_range : float
        Range for the covariate (uniformly sampled from [0, covariate_range]).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array with shape (n, 3): [time, status, X0]
    """
    if seed is not None:
        np.random.seed(seed)

    data = np.zeros((n, 3))

    for k in range(n):
        z = np.random.uniform(0, covariate_range)
        c = rfunc(1, cens_par)[0]
        x = np.random.exponential(scale=1 / np.exp(beta * z))

        time = min(x, c)
        status = int(x <= c)

        data[k, :] = [time, status, z]

    return data


def gen_cphm(
    n: int,
    model_cens: Literal["uniform", "exponential"],
    cens_par: float,
    beta: float,
    covariate_range: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate survival data following a Cox Proportional Hazards Model.

    Parameters
    ----------
    n : int
        Number of observations.
    model_cens : {"uniform", "exponential"}
        Type of censoring mechanism.
    cens_par : float
        Parameter for the censoring model.
    beta : float
        Coefficient for the covariate.
    covariate_range : float
        Upper bound for the covariate values (uniform between 0 and covariate_range).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["time", "status", "X0"]
        - time: observed event or censoring time
        - status: event indicator (1=event, 0=censored)
        - X0: predictor variable

    Examples
    --------
    >>> from gen_surv.cphm import gen_cphm
    >>> df = gen_cphm(n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
    >>> df.head()
       time  status        X0
    0  0.23     1.0       1.42
    1  0.78     0.0       0.89
    ...
    """
    validate_gen_cphm_inputs(n, model_cens, cens_par, covariate_range)

    rfunc = {"uniform": runifcens, "exponential": rexpocens}[model_cens]

    data = generate_cphm_data(n, rfunc, cens_par, beta, covariate_range, seed)
    return pd.DataFrame(data, columns=["time", "status", "X0"])
