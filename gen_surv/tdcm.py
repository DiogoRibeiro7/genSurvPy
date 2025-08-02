import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Sequence

from gen_surv.bivariate import sample_bivariate_distribution
from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validate import validate_gen_tdcm_inputs


def generate_censored_observations(
    n: int,
    dist_par: Sequence[float],
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    lam: float,
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Generate censored TDCM observations.

    Parameters:

        - n (int): Number of individuals
        - dist_par (list): Not directly used here (kept for API compatibility)
        - model_cens (str): "uniform" or "exponential"
        - cens_par (float): Parameter for the censoring model
        - beta (list): Length-2 list of regression coefficients
        - lam (float): Rate parameter
        - b (np.ndarray): Covariate matrix with 2 columns [., z1]

    Returns:
        - np.ndarray: Shape (n, 6) with columns:
          [id, start, stop, status, covariate1 (z1), covariate2 (z2)]
    """
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens

    z1 = b[:, 1]
    x = lam * b[:, 0] * np.exp(beta[0] * z1)
    u = np.random.uniform(size=n)
    c = rfunc(n, cens_par)

    threshold = 1 - np.exp(-x)
    exp_b0_z1 = np.exp(beta[0] * z1)
    log_term = -np.log(1 - u)
    t1 = log_term / (lam * exp_b0_z1)
    t2 = (log_term + x * (1 - np.exp(beta[1]))) / (
        lam * np.exp(beta[0] * z1 + beta[1])
    )
    mask = u < threshold
    t = np.where(mask, t1, t2)
    z2 = (~mask).astype(float)

    time = np.minimum(t, c)
    status = (t <= c).astype(float)

    ids = np.arange(1, n + 1, dtype=float)
    zeros = np.zeros(n, dtype=float)
    return np.column_stack((ids, zeros, time, status, z1, z2))


def gen_tdcm(
    n: int,
    dist: str,
    corr: float,
    dist_par: Sequence[float],
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    lam: float,
) -> pd.DataFrame:
    """
    Generate TDCM (Time-Dependent Covariate Model) survival data.

    Parameters:
    - n (int): Number of individuals.
    - dist (str): "weibull" or "exponential".
    - corr (float): Correlation coefficient.
    - dist_par (list): Distribution parameters.
    - model_cens (str): "uniform" or "exponential".
    - cens_par (float): Censoring parameter.
    - beta (list): Length-2 regression coefficients.
    - lam (float): Lambda rate parameter.

    Returns:
    - pd.DataFrame: Columns are ["id", "start", "stop", "status", "covariate", "tdcov"]
    """
    validate_gen_tdcm_inputs(n, dist, corr, dist_par, model_cens, cens_par, beta, lam)

    # Generate covariate matrix from bivariate distribution
    b = sample_bivariate_distribution(n, dist, corr, dist_par)

    data = generate_censored_observations(
        n, dist_par, model_cens, cens_par, beta, lam, b
    )

    return pd.DataFrame(
        data, columns=["id", "start", "stop", "status", "covariate", "tdcov"]
    )
