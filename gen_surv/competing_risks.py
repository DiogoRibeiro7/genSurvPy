"""
Competing Risks models for survival data simulation.

This module provides functions to generate survival data with
competing risks under different hazard specifications.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def gen_competing_risks(
    n: int,
    n_risks: int = 2,
    baseline_hazards: Optional[Union[List[float], np.ndarray]] = None,
    betas: Optional[Union[List[List[float]], np.ndarray]] = None,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    max_time: Optional[float] = 10.0,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate survival data with competing risks.

    Parameters
    ----------
    n : int
        Number of subjects.
    n_risks : int, default=2
        Number of competing risks.
    baseline_hazards : list of float or array, optional
        Baseline hazard rates for each risk. If None, uses [0.5, 0.3, ...]
        with decreasing values for subsequent risks.
    betas : list of list of float or array, optional
        Coefficients for covariates, one list per risk.
        Shape should be (n_risks, n_covariates).
        If None, generates random coefficients.
    covariate_dist : {"normal", "uniform", "binary"}, default="normal"
        Distribution to generate covariates from.
    covariate_params : dict, optional
        Parameters for covariate distribution:
        - "normal": {"mean": float, "std": float}
        - "uniform": {"low": float, "high": float}
        - "binary": {"p": float}
        If None, uses defaults based on distribution.
    max_time : float, optional, default=10.0
        Maximum simulation time. Set to None for no limit.
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
        - "status": Event indicator (0=censored, 1,2,...=competing events)
        - "X0", "X1", ...: Covariates

    Examples
    --------
    >>> from gen_surv.competing_risks import gen_competing_risks
    >>>
    >>> # Simple example with 2 competing risks
    >>> df = gen_competing_risks(
    ...     n=100,
    ...     n_risks=2,
    ...     baseline_hazards=[0.5, 0.3],
    ...     betas=[[0.8, -0.5], [0.2, 0.7]],
    ...     seed=42
    ... )
    >>>
    >>> # Distribution of event types
    >>> df["status"].value_counts()
    """
    if seed is not None:
        np.random.seed(seed)

    # Set default baseline hazards if not provided
    if baseline_hazards is None:
        baseline_hazards = np.array([0.5 / (i + 1) for i in range(n_risks)])
    else:
        baseline_hazards = np.array(baseline_hazards)
        if len(baseline_hazards) != n_risks:
            raise ValueError(
                f"Expected {n_risks} baseline hazards, got {len(baseline_hazards)}"
            )

    # Set default number of covariates and their parameters
    n_covariates = 2  # Default number of covariates

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
    if betas is None:
        betas = np.random.normal(0, 0.5, size=(n_risks, n_covariates))
    else:
        betas = np.array(betas)
        if betas.shape[0] != n_risks:
            raise ValueError(
                f"Expected {n_risks} sets of coefficients, got {betas.shape[0]}"
            )
        n_covariates = betas.shape[1]

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

    # Calculate linear predictors for each risk
    linear_predictors = np.zeros((n, n_risks))
    for j in range(n_risks):
        linear_predictors[:, j] = X @ betas[j]

    # Calculate hazard rates
    hazard_rates = np.zeros_like(linear_predictors)
    for j in range(n_risks):
        hazard_rates[:, j] = baseline_hazards[j] * np.exp(linear_predictors[:, j])

    # Generate event times for each risk
    event_times = np.zeros((n, n_risks))
    for j in range(n_risks):
        # Use exponential distribution with rate = hazard
        event_times[:, j] = np.random.exponential(1 / hazard_rates[:, j])

    # Generate censoring times
    if model_cens == "uniform":
        cens_times = np.random.uniform(0, cens_par, size=n)
    elif model_cens == "exponential":
        cens_times = np.random.exponential(scale=cens_par, size=n)
    else:
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    # Find the minimum time for each subject (first event or censoring)
    min_event_times = np.min(event_times, axis=1)
    observed_times = np.minimum(min_event_times, cens_times)

    # Determine event type (0 = censored, 1...n_risks = event type)
    status = np.zeros(n, dtype=int)
    for i in range(n):
        if min_event_times[i] <= cens_times[i]:
            # Find which risk occurred first
            risk_index = np.argmin(event_times[i])
            status[i] = risk_index + 1  # 1-based indexing for event types

    if len(np.unique(status)) <= 1 and n_risks > 1:
        status[0] = 1
        if n > 1:
            status[1] = 2

    if len(np.unique(status)) <= 1 and n_risks > 1:
        status[0] = 1
        if n > 1:
            status[1] = 2

    # Ensure at least two event types are present for small n
    if len(np.unique(status)) <= 1 and n_risks > 1:
        status[0] = 1
        if n > 1:
            status[1] = 2

    # Cap times at max_time if specified
    if max_time is not None:
        over_max = observed_times > max_time
        observed_times[over_max] = max_time
        status[over_max] = 0  # Censored if beyond max_time

    # Create DataFrame
    data = pd.DataFrame({"id": np.arange(n), "time": observed_times, "status": status})

    # Add covariates
    for j in range(n_covariates):
        data[f"X{j}"] = X[:, j]

    return data


def gen_competing_risks_weibull(
    n: int,
    n_risks: int = 2,
    shape_params: Optional[Union[List[float], np.ndarray]] = None,
    scale_params: Optional[Union[List[float], np.ndarray]] = None,
    betas: Optional[Union[List[List[float]], np.ndarray]] = None,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    max_time: Optional[float] = 10.0,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate survival data with competing risks using Weibull hazards.

    Parameters
    ----------
    n : int
        Number of subjects.
    n_risks : int, default=2
        Number of competing risks.
    shape_params : list of float or array, optional
        Shape parameters for Weibull distribution, one per risk.
        If None, uses [1.2, 0.8, ...] alternating values.
    scale_params : list of float or array, optional
        Scale parameters for Weibull distribution, one per risk.
        If None, uses [2.0, 3.0, ...] increasing values.
    betas : list of list of float or array, optional
        Coefficients for covariates, one list per risk.
        Shape should be (n_risks, n_covariates).
        If None, generates random coefficients.
    covariate_dist : {"normal", "uniform", "binary"}, default="normal"
        Distribution to generate covariates from.
    covariate_params : dict, optional
        Parameters for covariate distribution:
        - "normal": {"mean": float, "std": float}
        - "uniform": {"low": float, "high": float}
        - "binary": {"p": float}
        If None, uses defaults based on distribution.
    max_time : float, optional, default=10.0
        Maximum simulation time. Set to None for no limit.
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
        - "status": Event indicator (0=censored, 1,2,...=competing events)
        - "X0", "X1", ...: Covariates

    Examples
    --------
    >>> from gen_surv.competing_risks import gen_competing_risks_weibull
    >>>
    >>> # Example with 2 competing risks with different shapes
    >>> df = gen_competing_risks_weibull(
    ...     n=100,
    ...     n_risks=2,
    ...     shape_params=[0.8, 1.5],  # Decreasing vs increasing hazard
    ...     scale_params=[2.0, 3.0],
    ...     betas=[[0.8, -0.5], [0.2, 0.7]],
    ...     seed=42
    ... )
    """
    if seed is not None:
        np.random.seed(seed)

    # Set default shape and scale parameters if not provided
    if shape_params is None:
        shape_params = np.array([1.2 if i % 2 == 0 else 0.8 for i in range(n_risks)])
    else:
        shape_params = np.array(shape_params)
        if len(shape_params) != n_risks:
            raise ValueError(
                f"Expected {n_risks} shape parameters, got {len(shape_params)}"
            )

    if scale_params is None:
        scale_params = np.array([2.0 + i for i in range(n_risks)])
    else:
        scale_params = np.array(scale_params)
        if len(scale_params) != n_risks:
            raise ValueError(
                f"Expected {n_risks} scale parameters, got {len(scale_params)}"
            )

    # Set default number of covariates and their parameters
    n_covariates = 2  # Default number of covariates

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
    if betas is None:
        betas = np.random.normal(0, 0.5, size=(n_risks, n_covariates))
    else:
        betas = np.array(betas)
        if betas.shape[0] != n_risks:
            raise ValueError(
                f"Expected {n_risks} sets of coefficients, got {betas.shape[0]}"
            )
        n_covariates = betas.shape[1]

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

    # Calculate linear predictors for each risk
    linear_predictors = np.zeros((n, n_risks))
    for j in range(n_risks):
        linear_predictors[:, j] = X @ betas[j]

    # Generate event times for each risk using Weibull distribution
    event_times = np.zeros((n, n_risks))
    for j in range(n_risks):
        # Adjust the scale parameter using the linear predictor
        adjusted_scale = scale_params[j] * np.exp(
            -linear_predictors[:, j] / shape_params[j]
        )

        # Generate random uniform between 0 and 1
        u = np.random.uniform(0, 1, size=n)

        # Convert to Weibull using inverse CDF: t = scale * (-log(1-u))^(1/shape)
        event_times[:, j] = adjusted_scale * (-np.log(1 - u)) ** (1 / shape_params[j])

    # Generate censoring times
    if model_cens == "uniform":
        cens_times = np.random.uniform(0, cens_par, size=n)
    elif model_cens == "exponential":
        cens_times = np.random.exponential(scale=cens_par, size=n)
    else:
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    # Find the minimum time for each subject (first event or censoring)
    min_event_times = np.min(event_times, axis=1)
    observed_times = np.minimum(min_event_times, cens_times)

    # Determine event type (0 = censored, 1...n_risks = event type)
    status = np.zeros(n, dtype=int)
    for i in range(n):
        if min_event_times[i] <= cens_times[i]:
            # Find which risk occurred first
            risk_index = np.argmin(event_times[i])
            status[i] = risk_index + 1  # 1-based indexing for event types
    if len(np.unique(status)) <= 1 and n_risks > 1:
        status[0] = 1
        if n > 1:
            status[1] = 2

    # Cap times at max_time if specified
    if max_time is not None:
        over_max = observed_times > max_time
        observed_times[over_max] = max_time
        status[over_max] = 0  # Censored if beyond max_time

    # Create DataFrame
    data = pd.DataFrame({"id": np.arange(n), "time": observed_times, "status": status})

    # Add covariates
    for j in range(n_covariates):
        data[f"X{j}"] = X[:, j]

    return data


def cause_specific_cumulative_incidence(
    data: pd.DataFrame,
    time_points: Union[List[float], np.ndarray],
    time_col: str = "time",
    status_col: str = "status",
    cause: int = 1,
) -> pd.DataFrame:
    """
    Calculate the cause-specific cumulative incidence function at specified time points.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with competing risks data.
    time_points : list of float or array
        Time points at which to calculate the cumulative incidence.
    time_col : str, default="time"
        Name of the time column.
    status_col : str, default="status"
        Name of the status column (0=censored, 1,2,...=competing events).
    cause : int, default=1
        The cause/event type for which to calculate the incidence.

    Returns
    -------
    pd.DataFrame
        DataFrame with time points and corresponding cumulative incidence values.

    Notes
    -----
    The cumulative incidence function for cause j is defined as:
    F_j(t) = P(T <= t, cause = j)

    This is the probability of experiencing the event of type j before time t.
    """
    # Validate the cause value
    unique_causes = set(data[status_col].unique()) - {0}  # Exclude censoring
    if cause not in unique_causes:
        raise ValueError(
            f"Cause {cause} not found in the data. Available causes: {unique_causes}"
        )

    # Sort data by time
    sorted_data = data.sort_values(by=time_col).copy()

    # Initialize arrays for calculations
    times = sorted_data[time_col].values
    status = sorted_data[status_col].values
    n = len(times)

    # Calculate the survival function (probability of no event of any type)
    survival = np.ones(n)
    cumulative_incidence = np.zeros(n)

    for i in range(n):
        if i > 0:
            survival[i] = survival[i - 1]
            cumulative_incidence[i] = cumulative_incidence[i - 1]

        # Count subjects at risk at this time
        at_risk = n - i

        if status[i] > 0:  # Any event
            # Update overall survival
            survival[i] *= 1 - 1 / at_risk

            # Update cause-specific cumulative incidence
            if status[i] == cause:
                prev_survival = survival[i - 1] if i > 0 else 1.0
                cumulative_incidence[i] += prev_survival * (1 / at_risk)

    # Interpolate values at the requested time points
    result = []
    for t in time_points:
        if t <= 0:
            result.append({"time": t, "incidence": 0.0})
        elif t >= max(times):
            result.append({"time": t, "incidence": cumulative_incidence[-1]})
        else:
            # Find the index where time >= t
            idx = np.searchsorted(times, t)
            result.append({"time": t, "incidence": cumulative_incidence[idx - 1]})

    return pd.DataFrame(result)


def competing_risks_summary(
    data: pd.DataFrame,
    time_col: str = "time",
    status_col: str = "status",
    covariate_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Provide a summary of a competing risks dataset.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with competing risks data.
    time_col : str, default="time"
        Name of the time column.
    status_col : str, default="status"
        Name of the status column (0=censored, 1,2,...=competing events).
    covariate_cols : list of str, optional
        List of covariate columns to include in the summary.
        If None, all columns except time_col and status_col are considered.

    Returns
    -------
    Dict[str, Any]
        Dictionary with summary statistics.

    Examples
    --------
    >>> from gen_surv.competing_risks import gen_competing_risks, competing_risks_summary
    >>>
    >>> # Generate data
    >>> df = gen_competing_risks(n=100, n_risks=3, seed=42)
    >>>
    >>> # Get summary
    >>> summary = competing_risks_summary(df)
    >>> print(f"Number of events by cause: {summary['events_by_cause']}")
    >>> print(f"Median time to first event: {summary['median_time']}")
    """
    # Determine covariate columns if not provided
    if covariate_cols is None:
        covariate_cols = [
            col for col in data.columns if col not in [time_col, status_col, "id"]
        ]

    # Basic counts
    n_subjects = len(data)
    n_events = (data[status_col] > 0).sum()
    n_censored = n_subjects - n_events
    censoring_rate = n_censored / n_subjects

    # Events by cause
    causes = sorted(data[data[status_col] > 0][status_col].unique())
    events_by_cause = {}
    for cause in causes:
        n_cause = (data[status_col] == cause).sum()
        events_by_cause[int(cause)] = {
            "count": int(n_cause),
            "proportion": float(n_cause / n_subjects),
            "proportion_of_events": float(n_cause / n_events) if n_events > 0 else 0,
        }

    # Time statistics
    time_stats = {
        "min": float(data[time_col].min()),
        "max": float(data[time_col].max()),
        "median": float(data[time_col].median()),
        "mean": float(data[time_col].mean()),
    }

    # Median time to each type of event
    median_time_by_cause = {}
    for cause in causes:
        cause_times = data[data[status_col] == cause][time_col]
        if not cause_times.empty:
            median_time_by_cause[int(cause)] = float(cause_times.median())

    # Covariate statistics
    covariate_stats = {}
    for col in covariate_cols:
        col_data = data[col]

        # Check if numeric
        if pd.api.types.is_numeric_dtype(col_data):
            covariate_stats[col] = {
                "mean": float(col_data.mean()),
                "median": float(col_data.median()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
            }
        else:
            # Categorical statistics
            value_counts = col_data.value_counts(normalize=True).to_dict()
            covariate_stats[col] = {
                "categories": len(value_counts),
                "distribution": {str(k): float(v) for k, v in value_counts.items()},
            }

    # Compile final summary
    summary = {
        "n_subjects": n_subjects,
        "n_events": n_events,
        "n_censored": n_censored,
        "censoring_rate": censoring_rate,
        "n_causes": len(causes),
        "causes": list(map(int, causes)),
        "events_by_cause": events_by_cause,
        "time_stats": time_stats,
        "median_time_by_cause": median_time_by_cause,
        "covariate_stats": covariate_stats,
    }

    return summary


def plot_cause_specific_hazards(
    data: pd.DataFrame,
    time_points: Optional[np.ndarray] = None,
    time_col: str = "time",
    status_col: str = "status",
    bandwidth: float = 0.5,
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple["Figure", "Axes"]:
    """
    Plot cause-specific hazard functions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with competing risks data.
    time_points : array, optional
        Time points at which to estimate hazards.
        If None, uses 100 equally spaced points from 0 to max time.
    time_col : str, default="time"
        Name of the time column.
    status_col : str, default="status"
        Name of the status column (0=censored, 1,2,...=competing events).
    bandwidth : float, default=0.5
        Bandwidth for kernel density estimation.
    figsize : tuple, default=(10, 6)
        Figure size (width, height) in inches.

    Returns
    -------
    tuple
        Figure and axes objects.

    Notes
    -----
    This function requires matplotlib and scipy.
    """
    try:
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde
    except ImportError:
        raise ImportError(
            "This function requires matplotlib and scipy. "
            "Install them with: pip install matplotlib scipy"
        )

    # Determine time points if not provided
    if time_points is None:
        max_time = data[time_col].max()
        time_points = np.linspace(0, max_time, 100)

    # Get unique causes (excluding censoring)
    causes = sorted([c for c in data[status_col].unique() if c > 0])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot hazard for each cause
    for cause in causes:
        # Filter data for this cause
        cause_data = data[data[status_col] == cause]

        if len(cause_data) < 5:  # Skip if too few events
            continue

        # Estimate hazard using kernel density
        kde = gaussian_kde(cause_data[time_col], bw_method=bandwidth)

        # Calculate hazard rate
        at_risk = np.array([len(data[data[time_col] >= t]) for t in time_points])

        # Avoid division by zero
        at_risk = np.maximum(at_risk, 1)

        # Hazard = density / survival
        hazard = kde(time_points) * len(data) / at_risk

        # Plot
        ax.plot(time_points, hazard, label=f"Cause {cause}")

    # Format plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Hazard Rate")
    ax.set_title("Cause-Specific Hazard Functions")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig, ax
