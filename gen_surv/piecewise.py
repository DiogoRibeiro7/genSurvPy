"""
Piecewise Exponential survival models.

This module provides functions for generating survival data from piecewise
exponential distributions with time-dependent hazards.
"""

from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ._covariates import generate_covariates, prepare_betas, set_covariate_params
from .censoring import rexpocens, runifcens
from .validation import validate_gen_piecewise_inputs, validate_piecewise_params


def gen_piecewise_exponential(
    n: int,
    breakpoints: list[float],
    hazard_rates: list[float],
    betas: list[float] | NDArray[np.float64] | None = None,
    n_covariates: int = 2,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: dict[str, float] | None = None,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate survival data using a piecewise exponential distribution.

    Parameters
    ----------
    n : int
        Number of subjects.
    breakpoints : list of float
        Time points where hazard rates change. Must be in ascending order.
        The first interval is [0, breakpoints[0]), the second is [breakpoints[0], breakpoints[1]), etc.
    hazard_rates : list of float
        Hazard rates for each interval. Length should be len(breakpoints) + 1.
    betas : list or array, optional
        Coefficients for covariates. If None, generates random coefficients.
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
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "id": Subject identifier
        - "time": Time to event or censoring
        - "status": Event indicator (1=event, 0=censored)
        - "X0", "X1", ...: Covariates

    Examples
    --------
    >>> from gen_surv.piecewise import gen_piecewise_exponential
    >>>
    >>> # Generate data with 3 intervals (increasing hazard)
    >>> df = gen_piecewise_exponential(
    ...     n=100,
    ...     breakpoints=[1.0, 3.0],
    ...     hazard_rates=[0.2, 0.5, 1.0],
    ...     betas=[0.8, -0.5],
    ...     seed=42
    ... )
    """
    rng = np.random.default_rng(seed)

    validate_gen_piecewise_inputs(
        n,
        breakpoints,
        hazard_rates,
        n_covariates,
        model_cens,
        cens_par,
        covariate_dist,
    )
    covariate_params = set_covariate_params(covariate_dist, covariate_params)

    # Set default betas if not provided
    betas, n_covariates = prepare_betas(betas, n_covariates, rng)

    # Generate covariates
    X = generate_covariates(n, n_covariates, covariate_dist, covariate_params, rng)

    # Calculate linear predictor
    linear_predictor = X @ betas

    # Generate survival times using piecewise exponential distribution
    survival_times = np.zeros(n)

    for i in range(n):
        # Adjust hazard rates by the covariate effect
        adjusted_hazard_rates = [h * np.exp(linear_predictor[i]) for h in hazard_rates]

        # Generate random uniform between 0 and 1
        u = rng.uniform(0, 1)

        # Calculate survival time using inverse CDF method for piecewise exponential
        remaining_time = -np.log(u)  # Initial time remaining (for standard exponential)
        total_time = 0.0

        # Start with the first interval [0, breakpoints[0])
        interval_width = breakpoints[0]
        hazard = adjusted_hazard_rates[0]
        time_to_consume = remaining_time / hazard

        if time_to_consume < interval_width:
            # Event occurs in first interval
            survival_times[i] = time_to_consume
            continue

        # Event occurs after first interval
        total_time += interval_width
        remaining_time -= hazard * interval_width

        # Go through middle intervals [breakpoints[j-1], breakpoints[j])
        for j in range(1, len(breakpoints)):
            interval_width = breakpoints[j] - breakpoints[j - 1]
            hazard = adjusted_hazard_rates[j]
            time_to_consume = remaining_time / hazard

            if time_to_consume < interval_width:
                # Event occurs in this interval
                survival_times[i] = total_time + time_to_consume
                break

            # Event occurs after this interval
            total_time += interval_width
            remaining_time -= hazard * interval_width

        # If we've gone through all intervals and still no event,
        # use the last hazard rate for the remainder
        if remaining_time > 0:
            hazard = adjusted_hazard_rates[-1]
            survival_times[i] = total_time + remaining_time / hazard

    # Generate censoring times
    rfunc = runifcens if model_cens == "uniform" else rexpocens
    cens_times = rfunc(n, cens_par, rng)

    # Determine observed time and status
    observed_times = np.minimum(survival_times, cens_times)
    status = (survival_times <= cens_times).astype(int)

    # Create DataFrame
    data = pd.DataFrame({"id": np.arange(n), "time": observed_times, "status": status})

    # Add covariates
    for j in range(n_covariates):
        data[f"X{j}"] = X[:, j]

    return data


def piecewise_hazard_function(
    t: float | NDArray[np.float64],
    breakpoints: list[float],
    hazard_rates: list[float],
) -> float | NDArray[np.float64]:
    """
    Calculate the hazard function value at time t for a piecewise exponential distribution.

    Parameters
    ----------
    t : float or array
        Time point(s) at which to evaluate the hazard function.
    breakpoints : list of float
        Time points where hazard rates change.
    hazard_rates : list of float
        Hazard rates for each interval.

    Returns
    -------
    float or array
        Hazard function value(s) at time t.
    """
    validate_piecewise_params(breakpoints, hazard_rates)

    # Convert scalar input to array for consistent processing
    scalar_input = np.isscalar(t)
    t_array = np.atleast_1d(t)
    result = np.zeros_like(t_array)

    # Assign hazard rates based on time intervals
    result[t_array < 0] = 0  # Hazard is 0 for negative times

    # First interval: [0, breakpoints[0])
    mask = (t_array >= 0) & (t_array < breakpoints[0])
    result[mask] = hazard_rates[0]

    # Middle intervals: [breakpoints[j-1], breakpoints[j])
    for j in range(1, len(breakpoints)):
        mask = (t_array >= breakpoints[j - 1]) & (t_array < breakpoints[j])
        result[mask] = hazard_rates[j]

    # Last interval: [breakpoints[-1], infinity)
    mask = t_array >= breakpoints[-1]
    result[mask] = hazard_rates[-1]

    return result[0] if scalar_input else result


def piecewise_survival_function(
    t: float | NDArray[np.float64],
    breakpoints: list[float],
    hazard_rates: list[float],
) -> float | NDArray[np.float64]:
    """
    Calculate the survival function at time t for a piecewise exponential distribution.

    Parameters
    ----------
    t : float or array
        Time point(s) at which to evaluate the survival function.
    breakpoints : list of float
        Time points where hazard rates change.
    hazard_rates : list of float
        Hazard rates for each interval.

    Returns
    -------
    float or array
        Survival function value(s) at time t.
    """
    validate_piecewise_params(breakpoints, hazard_rates)

    # Convert scalar input to array for consistent processing
    scalar_input = np.isscalar(t)
    t_array = np.atleast_1d(t)
    result = np.ones_like(t_array)

    # For each time point, calculate the survival function
    for i, ti in enumerate(t_array):
        if ti <= 0:
            continue  # Survival probability is 1 at time 0 or earlier

        cumulative_hazard = 0.0

        # First interval: [0, min(ti, breakpoints[0]))
        first_interval_end = min(ti, breakpoints[0]) if breakpoints else ti
        cumulative_hazard += hazard_rates[0] * first_interval_end

        if ti <= breakpoints[0]:
            result[i] = np.exp(-cumulative_hazard)
            continue

        # Middle intervals: [breakpoints[j-1], min(ti, breakpoints[j]))
        for j in range(1, len(breakpoints)):
            if ti <= breakpoints[j - 1]:
                break

            interval_start = breakpoints[j - 1]
            interval_end = min(ti, breakpoints[j])
            interval_width = interval_end - interval_start

            cumulative_hazard += hazard_rates[j] * interval_width

            if ti <= breakpoints[j]:
                break

        # Last interval: [breakpoints[-1], ti)
        if ti > breakpoints[-1]:
            last_interval_width = ti - breakpoints[-1]
            cumulative_hazard += hazard_rates[-1] * last_interval_width

        result[i] = np.exp(-cumulative_hazard)

    return result[0] if scalar_input else result
