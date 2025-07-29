"""
Mixture Cure Models for survival data simulation.

This module provides functions to generate survival data with a cure fraction,
i.e., a proportion of subjects who are immune to the event of interest.
"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


def gen_mixture_cure(
    n: int,
    cure_fraction: float,
    baseline_hazard: float = 0.5,
    betas_survival: Optional[List[float]] = None,
    betas_cure: Optional[List[float]] = None,
    n_covariates: int = 2,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    max_time: Optional[float] = 10.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate survival data with a cure fraction using a mixture cure model.

    Parameters
    ----------
    n : int
        Number of subjects.
    cure_fraction : float
        Baseline probability of being cured (immune to the event).
        Should be between 0 and 1.
    baseline_hazard : float, default=0.5
        Baseline hazard rate for the non-cured population.
    betas_survival : list of float, optional
        Coefficients for covariates in the survival component.
        If None, generates random coefficients.
    betas_cure : list of float, optional
        Coefficients for covariates in the cure component.
        If None, generates random coefficients.
    n_covariates : int, default=2
        Number of covariates to generate if betas is None.
    covariate_dist : {"normal", "uniform", "binary"}, default="normal"
        Distribution to generate covariates from.
    covariate_params : dict, optional
        Parameters for covariate distribution:
        - "normal": {"mean": float, "std": float}
        - "uniform": {"low": float, "high": float}
        - "binary": {"p": float}
        If None, uses defaults based on distribution.
    model_cens : {"uniform", "exponential"}, default="uniform"
        Censoring mechanism.
    cens_par : float, default=5.0
        Parameter for censoring distribution.
    max_time : float, optional, default=10.0
        Maximum simulation time. Set to None for no limit.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "id": Subject identifier
        - "time": Time to event or censoring
        - "status": Event indicator (1=event, 0=censored)
        - "cured": Indicator of cure status (1=cured, 0=not cured)
        - "X0", "X1", ...: Covariates

    Examples
    --------
    >>> from gen_surv.mixture import gen_mixture_cure
    >>>
    >>> # Generate data with 30% baseline cure fraction
    >>> df = gen_mixture_cure(
    ...     n=100,
    ...     cure_fraction=0.3,
    ...     betas_survival=[0.8, -0.5],
    ...     betas_cure=[-0.5, 0.8],
    ...     seed=42
    ... )
    >>>
    >>> # Check cure proportion
    >>> print(f"Cured subjects: {df['cured'].mean():.2%}")
    """
    if seed is not None:
        np.random.seed(seed)

    # Validate inputs
    if not 0 <= cure_fraction <= 1:
        raise ValueError("cure_fraction must be between 0 and 1")

    if baseline_hazard <= 0:
        raise ValueError("baseline_hazard must be positive")

    # Set default covariate parameters if not provided
    if covariate_params is None:
        if covariate_dist == "normal":
            covariate_params = {"mean": 0.0, "std": 1.0}
        elif covariate_dist == "uniform":
            covariate_params = {"low": 0.0, "high": 1.0}
        elif covariate_dist == "binary":
            covariate_params = {"p": 0.5}
        else:
            raise ValueError(f"Unknown covariate distribution: {covariate_dist}")

    # Set default betas if not provided
    if betas_survival is None:
        betas_survival = np.random.normal(0, 0.5, size=n_covariates)
    else:
        betas_survival = np.array(betas_survival)
        n_covariates = len(betas_survival)

    if betas_cure is None:
        betas_cure = np.random.normal(0, 0.5, size=n_covariates)
    else:
        betas_cure = np.array(betas_cure)
        if len(betas_cure) != n_covariates:
            raise ValueError(
                f"betas_cure must have the same length as betas_survival, "
                f"got {len(betas_cure)} vs {n_covariates}"
            )

    # Generate covariates
    if covariate_dist == "normal":
        X = np.random.normal(
            covariate_params.get("mean", 0.0),
            covariate_params.get("std", 1.0),
            size=(n, n_covariates),
        )
    elif covariate_dist == "uniform":
        X = np.random.uniform(
            covariate_params.get("low", 0.0),
            covariate_params.get("high", 1.0),
            size=(n, n_covariates),
        )
    elif covariate_dist == "binary":
        X = np.random.binomial(
            1, covariate_params.get("p", 0.5), size=(n, n_covariates)
        )
    else:
        raise ValueError(f"Unknown covariate distribution: {covariate_dist}")

    # Calculate linear predictors
    lp_survival = X @ betas_survival
    lp_cure = X @ betas_cure

    # Determine cure status (logistic model)
    cure_probs = 1 / (
        1 + np.exp(-(np.log(cure_fraction / (1 - cure_fraction)) + lp_cure))
    )
    cured = np.random.binomial(1, cure_probs)

    # Generate survival times
    survival_times = np.zeros(n)

    # For non-cured subjects, generate event times
    non_cured_indices = np.where(cured == 0)[0]

    for i in non_cured_indices:
        # Adjust hazard rate by covariate effect
        adjusted_hazard = baseline_hazard * np.exp(lp_survival[i])

        # Generate exponential survival time
        survival_times[i] = np.random.exponential(scale=1 / adjusted_hazard)

    # For cured subjects, set "infinite" survival time
    cured_indices = np.where(cured == 1)[0]
    if max_time is not None:
        survival_times[cured_indices] = max_time * 100  # Effectively infinite
    else:
        survival_times[cured_indices] = np.inf  # Actually infinite

    # Generate censoring times
    if model_cens == "uniform":
        cens_times = np.random.uniform(0, cens_par, size=n)
    elif model_cens == "exponential":
        cens_times = np.random.exponential(scale=cens_par, size=n)
    else:
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    # Determine observed time and status
    observed_times = np.minimum(survival_times, cens_times)
    status = (survival_times <= cens_times).astype(int)

    # Cap times at max_time if specified
    if max_time is not None:
        over_max = observed_times > max_time
        observed_times[over_max] = max_time
        status[over_max] = 0  # Censored if beyond max_time

    # Create DataFrame
    data = pd.DataFrame(
        {"id": np.arange(n), "time": observed_times, "status": status, "cured": cured}
    )

    # Add covariates
    for j in range(n_covariates):
        data[f"X{j}"] = X[:, j]

    return data


def cure_fraction_estimate(
    data: pd.DataFrame,
    time_col: str = "time",
    status_col: str = "status",
    bandwidth: float = 0.1,
) -> float:
    """
    Estimate the cure fraction from observed data using non-parametric methods.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with survival data.
    time_col : str, default="time"
        Name of the time column.
    status_col : str, default="status"
        Name of the status column (1=event, 0=censored).
    bandwidth : float, default=0.1
        Bandwidth parameter for smoothing the tail of the survival curve.

    Returns
    -------
    float
        Estimated cure fraction.

    Notes
    -----
    This function uses a non-parametric approach to estimate the cure fraction
    based on the plateau of the survival curve. It may not be accurate for
    small sample sizes or heavy censoring.
    """
    # Sort data by time
    sorted_data = data.sort_values(by=time_col).copy()

    # Calculate Kaplan-Meier estimate
    times = sorted_data[time_col].values
    status = sorted_data[status_col].values
    n = len(times)

    if n == 0:
        return 0.0

    # Calculate survival function
    survival = np.ones(n)

    for i in range(n):
        if i > 0:
            survival[i] = survival[i - 1]

        # Count subjects at risk at this time
        at_risk = n - i

        if status[i] == 1:  # Event
            survival[i] *= 1 - 1 / at_risk

    # Estimate cure fraction as the plateau of the survival curve
    # Use the last 10% of the survival curve if enough data points
    tail_size = max(int(n * 0.1), 1)
    tail_survival = survival[-tail_size:]

    # Apply smoothing if there are enough data points
    if tail_size > 3:
        # Use kernel smoothing
        weights = np.exp(
            -((np.arange(tail_size) - tail_size + 1) ** 2)
            / (2 * bandwidth * tail_size) ** 2
        )
        weights = weights / weights.sum()
        cure_fraction = np.sum(tail_survival * weights)
    else:
        # Just use the last survival probability
        cure_fraction = survival[-1]

    return cure_fraction
