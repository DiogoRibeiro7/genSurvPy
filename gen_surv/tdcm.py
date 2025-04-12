import numpy as np
import pandas as pd
from gen_surv.validate import validate_gen_tdcm_inputs
from gen_surv.censoring import runifcens, rexpocens

def generate_censored_observations(n, dist_par, model_cens, cens_par, beta, lam, b):
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
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    observations = np.zeros((n, 6))

    for k in range(n):
        z1 = b[k, 1]
        c = rfunc(1, cens_par)[0]
        u = np.random.uniform()

        # Determine path based on u threshold
        threshold = 1 - np.exp(-lam * b[k, 0] * np.exp(beta[0] * z1))
        if u < threshold:
            t = -np.log(1 - u) / (lam * np.exp(beta[0] * z1))
            z2 = 0
        else:
            t = (
                -np.log(1 - u)
                + lam * b[k, 0] * np.exp(beta[0] * z1) * (1 - np.exp(beta[2]))
            ) / (lam * np.exp(beta[0] * z1 + beta[1]))
            z2 = 1

        time = min(t, c)
        status = int(t <= c)

        observations[k] = [k + 1, 0, time, status, z1, z2]

    return observations


def gen_tdcm(n, dist, corr, dist_par, model_cens, cens_par, beta, lam):
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

    # Generate covariate matrix with correlation structure
    mean = [0, 0]
    cov_matrix = [[1, corr], [corr, 1]]
    b = np.random.multivariate_normal(mean, cov_matrix, size=n)

    data = generate_censored_observations(n, dist_par, model_cens, cens_par, beta, lam, b)

    return pd.DataFrame(data, columns=["id", "start", "stop", "status", "covariate", "tdcov"])
