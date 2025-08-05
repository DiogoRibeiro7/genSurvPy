"""Utilities for generating covariate matrices with validation."""

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from ._validation import ParameterError, ensure_positive

_CovParams = dict[str, float | tuple[float, float]]


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


def generate_covariates(
    n: int,
    n_covariates: int,
    covariate_dist: Literal["normal", "uniform", "binary"],
    covariate_params: _CovParams,
) -> NDArray[np.float64]:
    """Generate covariate matrix according to the specified distribution."""
    if covariate_dist == "normal":
        std = cast(float, covariate_params.get("std", 1.0))
        ensure_positive(std, "covariate_params['std']")
        mean = cast(float, covariate_params.get("mean", 0.0))
        return np.random.normal(mean, std, size=(n, n_covariates))
    if covariate_dist == "uniform":
        low = cast(float, covariate_params.get("low", 0.0))
        high = cast(float, covariate_params.get("high", 1.0))
        if high <= low:
            raise ParameterError(
                "covariate_params['high']",
                high,
                "must be greater than 'low'",
            )
        return np.random.uniform(low, high, size=(n, n_covariates))
    if covariate_dist == "binary":
        p = cast(float, covariate_params.get("p", 0.5))
        if not 0 <= p <= 1:
            raise ParameterError(
                "covariate_params['p']",
                p,
                "must be between 0 and 1",
            )
        return np.random.binomial(1, p, size=(n, n_covariates)).astype(float)
    raise ParameterError(
        "covariate_dist",
        covariate_dist,
        "unsupported covariate distribution; choose from 'normal', 'uniform', or 'binary'",
    )
