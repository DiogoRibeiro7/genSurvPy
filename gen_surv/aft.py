import numpy as np
import pandas as pd


def gen_aft_log_normal(n, beta, sigma, model_cens, cens_par, seed=None):
    """
    Simulate survival data under a Log-Normal Accelerated Failure Time (AFT) model.

    Parameters:
    - n (int): Number of individuals
    - beta (list of float): Coefficients for covariates
    - sigma (float): Standard deviation of the log-error term
    - model_cens (str): 'uniform' or 'exponential'
    - cens_par (float): Parameter for censoring distribution
    - seed (int, optional): Random seed

    Returns:
    - pd.DataFrame: DataFrame with columns ['id', 'time', 'status', 'X0', ..., 'Xp']
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(beta)
    X = np.random.normal(size=(n, p))
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
    log_T = X @ np.array(beta) + epsilon
    T = np.exp(log_T)

    if model_cens == "uniform":
        C = np.random.uniform(0, cens_par, size=n)
    elif model_cens == "exponential":
        C = np.random.exponential(scale=cens_par, size=n)
    else:
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    observed_time = np.minimum(T, C)
    status = (T <= C).astype(int)

    data = pd.DataFrame({
        "id": np.arange(n),
        "time": observed_time,
        "status": status
    })

    for j in range(p):
        data[f"X{j}"] = X[:, j]

    return data
