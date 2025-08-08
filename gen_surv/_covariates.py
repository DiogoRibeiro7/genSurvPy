"""Utilities for generating covariate matrices with validation."""

from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .validation import ParameterError, ensure_positive

_CovParams = dict[str, float]


def set_covariate_params(
    covariate_dist: Literal["normal", "uniform", "binary"],
    covariate_params: _CovParams | None,
) -> _CovParams:
    """Return covariate distribution parameters with defaults filled in."""
    if covariate_params is not None:
        return covariate_params
    if covariate_dist == "normal":
        return {"mean": 0.0, "std": 1.0}
    if covariate_dist == "uniform":
        return {"low": 0.0, "high": 1.0}
    if covariate_dist == "binary":
        return {"p": 0.5}
    raise ParameterError(
        "covariate_dist",
        covariate_dist,
        "unsupported covariate distribution; choose from 'normal', 'uniform', or 'binary'",
    )


def _get_float(params: _CovParams, key: str, default: float) -> float:
    val = params.get(key, default)
    if not isinstance(val, (int, float)):
        raise ParameterError(f"covariate_params['{key}']", val, "must be a number")
    return float(val)


def generate_covariates(
    n: int,
    n_covariates: int,
    covariate_dist: Literal["normal", "uniform", "binary"],
    covariate_params: _CovParams,
    rng: Generator,
) -> NDArray[np.float64]:
    """Generate covariate matrix according to the specified distribution."""
    if covariate_dist == "normal":
        std = _get_float(covariate_params, "std", 1.0)
        ensure_positive(std, "covariate_params['std']")
        mean = _get_float(covariate_params, "mean", 0.0)
        return rng.normal(mean, std, size=(n, n_covariates))
    if covariate_dist == "uniform":
        low = _get_float(covariate_params, "low", 0.0)
        high = _get_float(covariate_params, "high", 1.0)
        if high <= low:
            raise ParameterError(
                "covariate_params['high']",
                high,
                "must be greater than 'low'",
            )
        return rng.uniform(low, high, size=(n, n_covariates))
    if covariate_dist == "binary":
        p = _get_float(covariate_params, "p", 0.5)
        if not 0 <= p <= 1:
            raise ParameterError(
                "covariate_params['p']",
                p,
                "must be between 0 and 1",
            )
        return rng.binomial(1, p, size=(n, n_covariates)).astype(float)
    raise ParameterError(
        "covariate_dist",
        covariate_dist,
        "unsupported covariate distribution; choose from 'normal', 'uniform', or 'binary'",
    )
