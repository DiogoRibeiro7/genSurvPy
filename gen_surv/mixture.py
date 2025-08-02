"""
Mixture Cure Models for survival data simulation.

This module provides functions to generate survival data with a cure fraction,
i.e., a proportion of subjects who are immune to the event of interest.
"""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray


_TAIL_FRACTION = 0.1
_SMOOTH_MIN_TAIL = 3

from ._validation import (
    ensure_censoring_model,
    ensure_in_choices,
    ensure_positive,
    LengthError,
    ParameterError,
)
from .censoring import rexpocens, runifcens


def _set_covariate_params(
    covariate_dist: str,
    covariate_params: dict[str, float | tuple[float, float]] | None,
) -> dict[str, float | tuple[float, float]]:
    if covariate_params is not None:
        return covariate_params
    if covariate_dist == "normal":
        return {"mean": 0.0, "std": 1.0}
    if covariate_dist == "uniform":
        return {"low": 0.0, "high": 1.0}
    if covariate_dist == "binary":
        return {"p": 0.5}
    raise ParameterError(
        "covariate_dist", covariate_dist, "must be one of {'normal','uniform','binary'}"
    )


def _prepare_betas(
    betas_survival: list[float] | None,
    betas_cure: list[float] | None,
    n_covariates: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], int]:
    if betas_survival is None:
        betas_survival_arr = np.random.normal(0, 0.5, size=n_covariates)
    else:
        betas_survival_arr = np.asarray(betas_survival, dtype=float)
        n_covariates = len(betas_survival_arr)

    if betas_cure is None:
        betas_cure_arr = np.random.normal(0, 0.5, size=n_covariates)
    else:
        betas_cure_arr = np.asarray(betas_cure, dtype=float)
        if len(betas_cure_arr) != n_covariates:
            raise LengthError("betas_cure", len(betas_cure_arr), n_covariates)

    return betas_survival_arr, betas_cure_arr, n_covariates


def _generate_covariates(
    n: int,
    n_covariates: int,
    covariate_dist: str,
    covariate_params: dict[str, float | tuple[float, float]],
) -> NDArray[np.float64]:
    if covariate_dist == "normal":
        return np.random.normal(
            covariate_params.get("mean", 0.0),
            covariate_params.get("std", 1.0),
            size=(n, n_covariates),
        )
    if covariate_dist == "uniform":
        return np.random.uniform(
            covariate_params.get("low", 0.0),
            covariate_params.get("high", 1.0),
            size=(n, n_covariates),
        )
    if covariate_dist == "binary":
        return np.random.binomial(
            1, covariate_params.get("p", 0.5), size=(n, n_covariates)
        ).astype(float)
    raise ParameterError(
        "covariate_dist", covariate_dist, "must be one of {'normal','uniform','binary'}"
    )


def _cure_status(
    lp_cure: NDArray[np.float64], cure_fraction: float
) -> NDArray[np.int64]:
    cure_probs = 1 / (
        1 + np.exp(-(np.log(cure_fraction / (1 - cure_fraction)) + lp_cure))
    )
    return np.random.binomial(1, cure_probs).astype(np.int64)


def _survival_times(
    cured: NDArray[np.int64],
    lp_survival: NDArray[np.float64],
    baseline_hazard: float,
    max_time: float | None,
) -> NDArray[np.float64]:
    n = cured.size
    times = np.zeros(n, dtype=float)
    non_cured = cured == 0
    adjusted_hazard = baseline_hazard * np.exp(lp_survival[non_cured])
    times[non_cured] = np.random.exponential(scale=1 / adjusted_hazard)
    if max_time is not None:
        times[~non_cured] = max_time * 100
    else:
        times[~non_cured] = np.inf
    return times


def _apply_censoring(
    survival_times: NDArray[np.float64],
    model_cens: str,
    cens_par: float,
    max_time: float | None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    cens_times = rfunc(len(survival_times), cens_par)
    observed = np.minimum(survival_times, cens_times)
    status = (survival_times <= cens_times).astype(int)
    if max_time is not None:
        over_max = observed > max_time
        observed[over_max] = max_time
        status[over_max] = 0
    return observed, status


def gen_mixture_cure(
    n: int,
    cure_fraction: float,
    baseline_hazard: float = 0.5,
    betas_survival: list[float] | None = None,
    betas_cure: list[float] | None = None,
    n_covariates: int = 2,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: dict[str, float | tuple[float, float]] | None = None,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    max_time: float | None = 10.0,
    seed: int | None = None,
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

    if not 0 <= cure_fraction <= 1:
        raise ParameterError(
            "cure_fraction", cure_fraction, "must be between 0 and 1"
        )
    ensure_positive(baseline_hazard, "baseline_hazard")

    ensure_in_choices(
        covariate_dist, "covariate_dist", {"normal", "uniform", "binary"}
    )
    covariate_params = _set_covariate_params(covariate_dist, covariate_params)
    betas_survival_arr, betas_cure_arr, n_covariates = _prepare_betas(
        betas_survival, betas_cure, n_covariates
    )
    X = _generate_covariates(n, n_covariates, covariate_dist, covariate_params)
    lp_survival = X @ betas_survival_arr
    lp_cure = X @ betas_cure_arr
    cured = _cure_status(lp_cure, cure_fraction)
    survival_times = _survival_times(cured, lp_survival, baseline_hazard, max_time)

    ensure_censoring_model(model_cens)
    observed_times, status = _apply_censoring(
        survival_times, model_cens, cens_par, max_time
    )

    data = pd.DataFrame(
        {"id": np.arange(n), "time": observed_times, "status": status, "cured": cured}
    )

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
    # Use the last portion of the survival curve if enough data points
    tail_size = max(int(n * _TAIL_FRACTION), 1)
    tail_survival = survival[-tail_size:]

    # Apply smoothing if there are enough data points
    if tail_size > _SMOOTH_MIN_TAIL:
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
