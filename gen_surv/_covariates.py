"""Utilities for generating covariate matrices with validation."""

from typing import Literal, Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .validation import (
    LengthError,
    ListOfListsError,
    NumericSequenceError,
    ParameterError,
    ensure_numeric_sequence,
    ensure_positive,
    ensure_sequence_length,
)

_CovParams = dict[str, float]


def set_covariate_params(
    covariate_dist: Literal["normal", "uniform", "binary"],
    covariate_params: _CovParams | None,
) -> _CovParams:
    """Return covariate distribution parameters with defaults filled in.

    Parameters
    ----------
    covariate_dist : {"normal", "uniform", "binary"}
        Distribution used to sample covariates.
    covariate_params : dict[str, float], optional
        Parameters specific to the chosen distribution. Missing keys are
        populated with sensible defaults.

    Returns
    -------
    dict[str, float]
        Completed parameter dictionary for ``covariate_dist``.
    """
    if covariate_dist == "normal":
        if covariate_params is None:
            return {"mean": 0.0, "std": 1.0}
        if {"mean", "std"} <= covariate_params.keys():
            return covariate_params
        raise ParameterError(
            "covariate_params",
            covariate_params,
            "must include 'mean' and 'std'",
        )
    if covariate_dist == "uniform":
        if covariate_params is None:
            return {"low": 0.0, "high": 1.0}
        if {"low", "high"} <= covariate_params.keys():
            return covariate_params
        raise ParameterError(
            "covariate_params",
            covariate_params,
            "must include 'low' and 'high'",
        )
    if covariate_dist == "binary":
        if covariate_params is None:
            return {"p": 0.5}
        if "p" in covariate_params:
            return covariate_params
        raise ParameterError("covariate_params", covariate_params, "must include 'p'")
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
    """Generate covariate matrix according to the specified distribution.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    n_covariates : int
        Number of covariate columns.
    covariate_dist : {"normal", "uniform", "binary"}
        Distribution used to sample covariates.
    covariate_params : dict[str, float]
        Parameters specific to ``covariate_dist``.
    rng : Generator
        Random number generator used for sampling.

    Returns
    -------
    NDArray[np.float64]
        Matrix of shape ``(n, n_covariates)`` containing sampled covariates.
    """
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


def prepare_betas(
    betas: Sequence[float] | None,
    n_covariates: int,
    rng: Generator,
    *,
    name: str = "betas",
    enforce_length: bool = False,
) -> tuple[NDArray[np.float64], int]:
    """Return coefficient array, generating defaults when needed.

    Parameters
    ----------
    betas : sequence of float, optional
        Coefficient values. If ``None``, random values are generated.
    n_covariates : int
        Expected number of coefficients when generating defaults.
    rng : Generator
        Random number generator used when ``betas`` is ``None``.
    name : str, optional
        Name used in error messages.
    enforce_length : bool, optional
        If ``True``, raise an error when ``betas`` does not have exactly
        ``n_covariates`` elements.

    Returns
    -------
    tuple[NDArray[np.float64], int]
        A tuple containing the coefficient array and the number of
        covariates represented by it.
    """
    if betas is None:
        return rng.normal(0, 0.5, size=n_covariates), n_covariates
    ensure_numeric_sequence(betas, name)
    arr = np.asarray(betas, dtype=float)
    if enforce_length and len(arr) != n_covariates:
        raise LengthError(name, len(arr), n_covariates)
    return arr, len(arr)


def prepare_betas_matrix(
    betas: Sequence[Sequence[float]] | None,
    n_risks: int,
    n_covariates: int,
    rng: Generator,
    *,
    name: str = "betas",
) -> tuple[NDArray[np.float64], int]:
    """Return coefficient matrix for multiple risks.

    Parameters
    ----------
    betas : sequence of sequence of float, optional
        Coefficient matrix where each sub-sequence corresponds to a risk.
        Random values are generated when ``None``.
    n_risks : int
        Number of competing risks.
    n_covariates : int
        Number of covariates per risk.
    rng : Generator
        Random number generator used when ``betas`` is ``None``.
    name : str, optional
        Name used in error messages.

    Returns
    -------
    tuple[NDArray[np.float64], int]
        A tuple containing the coefficient matrix of shape
        ``(n_risks, n_covariates)`` and the number of covariates.
    """
    if betas is None:
        return rng.normal(0, 0.5, size=(n_risks, n_covariates)), n_covariates
    if not isinstance(betas, Sequence) or any(
        not isinstance(b, Sequence) for b in betas
    ):
        raise ListOfListsError(name, betas)
    arr = np.asarray(betas, dtype=float)
    ensure_sequence_length(arr, n_risks, name)
    for j in range(n_risks):
        ensure_numeric_sequence(arr[j], f"{name}[{j}]")
        nonfinite = np.where(~np.isfinite(arr[j]))[0]
        if nonfinite.size:
            idx = int(nonfinite[0])
            raise NumericSequenceError(f"{name}[{j}]", arr[j][idx], idx)
    return arr, arr.shape[1]
