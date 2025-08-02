"""
Accelerated Failure Time (AFT) models including Weibull, Log-Normal, and Log-Logistic distributions.
"""

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from ._validation import ensure_censoring_model, ensure_positive
from .censoring import rexpocens, runifcens


def gen_aft_log_normal(
    n: int,
    beta: List[float],
    sigma: float,
    model_cens: Literal["uniform", "exponential"],
    cens_par: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate survival data under a Log-Normal Accelerated Failure Time (AFT) model.

    Parameters
    ----------
    n : int
        Number of individuals
    beta : list of float
        Coefficients for covariates
    sigma : float
        Standard deviation of the log-error term
    model_cens : {"uniform", "exponential"}
        Censoring mechanism
    cens_par : float
        Parameter for censoring distribution
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['id', 'time', 'status', 'X0', ..., 'Xp']
    """
    if seed is not None:
        np.random.seed(seed)

    p = len(beta)
    X = np.random.normal(size=(n, p))
    epsilon = np.random.normal(loc=0.0, scale=sigma, size=n)
    log_T = X @ np.array(beta) + epsilon
    T = np.exp(log_T)

    ensure_censoring_model(model_cens)
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    C = rfunc(n, cens_par)

    observed_time = np.minimum(T, C)
    status = (T <= C).astype(int)

    data = pd.DataFrame({"id": np.arange(n), "time": observed_time, "status": status})

    for j in range(p):
        data[f"X{j}"] = X[:, j]

    return data


def gen_aft_weibull(
    n: int,
    beta: List[float],
    shape: float,
    scale: float,
    model_cens: Literal["uniform", "exponential"],
    cens_par: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate survival data under a Weibull Accelerated Failure Time (AFT) model.

    The Weibull AFT model has survival function:
    S(t|X) = exp(-(t/scale)^shape * exp(-X*beta))

    Parameters
    ----------
    n : int
        Number of individuals
    beta : list of float
        Coefficients for covariates
    shape : float
        Weibull shape parameter (k > 0)
    scale : float
        Weibull scale parameter (λ > 0)
    model_cens : {"uniform", "exponential"}
        Censoring mechanism
    cens_par : float
        Parameter for censoring distribution
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['id', 'time', 'status', 'X0', ..., 'Xp']
    """
    if seed is not None:
        np.random.seed(seed)

    ensure_positive(shape, "shape")
    ensure_positive(scale, "scale")

    p = len(beta)
    X = np.random.normal(size=(n, p))

    # Linear predictor
    eta = X @ np.array(beta)

    # Generate Weibull survival times
    U = np.random.uniform(size=n)
    T = scale * (-np.log(U) * np.exp(-eta)) ** (1 / shape)

    # Generate censoring times
    ensure_censoring_model(model_cens)
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    C = rfunc(n, cens_par)

    # Observed time is the minimum of event time and censoring time
    observed_time = np.minimum(T, C)
    status = (T <= C).astype(int)

    data = pd.DataFrame({"id": np.arange(n), "time": observed_time, "status": status})

    for j in range(p):
        data[f"X{j}"] = X[:, j]

    return data


def gen_aft_log_logistic(
    n: int,
    beta: List[float],
    shape: float,
    scale: float,
    model_cens: Literal["uniform", "exponential"],
    cens_par: float,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate survival data under a Log-Logistic Accelerated Failure Time (AFT) model.

    The Log-Logistic AFT model has survival function:
    S(t|X) = 1 / (1 + (t/scale)^shape * exp(X*beta))

    Log-logistic distribution is useful when the hazard rate first increases and then decreases.

    Parameters
    ----------
    n : int
        Number of individuals
    beta : list of float
        Coefficients for covariates
    shape : float
        Log-logistic shape parameter (α > 0)
    scale : float
        Log-logistic scale parameter (β > 0)
    model_cens : {"uniform", "exponential"}
        Censoring mechanism
    cens_par : float
        Parameter for censoring distribution
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['id', 'time', 'status', 'X0', ..., 'Xp']
    """
    if seed is not None:
        np.random.seed(seed)

    ensure_positive(shape, "shape")
    ensure_positive(scale, "scale")

    p = len(beta)
    X = np.random.normal(size=(n, p))

    # Linear predictor
    eta = X @ np.array(beta)

    # Generate Log-Logistic survival times
    U = np.random.uniform(size=n)

    # Inverse CDF method: S(t) = 1/(1 + (t/scale)^shape)
    # so t = scale * (1/S - 1)^(1/shape)
    # For random U ~ Uniform(0,1), we can use U as 1-S
    # t = scale * (1/(1-U) - 1)^(1/shape) * exp(-eta/shape)
    # simplifies to: t = scale * (U/(1-U))^(1/shape) * exp(-eta/shape)

    # Avoid numerical issues near 1
    U = np.clip(U, 0.001, 0.999)
    T = scale * (U / (1 - U)) ** (1 / shape) * np.exp(-eta / shape)

    # Generate censoring times
    ensure_censoring_model(model_cens)
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    C = rfunc(n, cens_par)

    # Observed time is the minimum of event time and censoring time
    observed_time = np.minimum(T, C)
    status = (T <= C).astype(int)

    data = pd.DataFrame({"id": np.arange(n), "time": observed_time, "status": status})

    for j in range(p):
        data[f"X{j}"] = X[:, j]

    return data
