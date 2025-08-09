"""Input validation utilities.

This module unifies the low-level validation helpers and the higher-level
checks used by the data generators.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Real
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray


class ValidationError(ValueError):
    """Base class for input validation errors."""


class PositiveIntegerError(ValidationError):
    """Raised when a value expected to be a positive integer is invalid."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(
            f"Argument '{name}' must be a positive integer; got {value!r} of type {type(value).__name__}. "
            "Please provide a whole number greater than 0."
        )


class PositiveValueError(ValidationError):
    """Raised when a value expected to be positive is invalid."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(
            f"Argument '{name}' must be greater than 0; got {value!r} of type {type(value).__name__}. "
            "Try a positive number such as 1.0."
        )


class ChoiceError(ValidationError):
    """Raised when a value is not among an allowed set of choices."""

    def __init__(self, name: str, value: Any, choices: Iterable[str]) -> None:
        choices_str = "', '".join(sorted(choices))
        super().__init__(
            f"Argument '{name}' must be one of '{choices_str}'; got {value!r} of type {type(value).__name__}. "
            "Choose a valid option."
        )


class LengthError(ValidationError):
    """Raised when a sequence does not have the expected length."""

    def __init__(self, name: str, actual: int, expected: int) -> None:
        super().__init__(
            f"Argument '{name}' must be a sequence of length {expected}; got length {actual}. "
            "Adjust the number of elements."
        )


class NumericSequenceError(ValidationError):
    """Raised when a sequence contains non-numeric elements."""

    def __init__(self, name: str, value: Any, index: int | None = None) -> None:
        if index is None:
            super().__init__(
                f"All elements in '{name}' must be numeric; got {value!r}. "
                "Convert or remove non-numeric values."
            )
        else:
            super().__init__(
                f"All elements in '{name}' must be numeric; found {value!r} of type {type(value).__name__} at index {index}. "
                "Replace or remove this entry."
            )


class PositiveSequenceError(ValidationError):
    """Raised when a sequence contains non-positive elements."""

    def __init__(self, name: str, value: Any, index: int) -> None:
        super().__init__(
            f"All elements in '{name}' must be greater than 0; found {value!r} at index {index}. "
            "Use positive numbers only."
        )


class ListOfListsError(ValidationError):
    """Raised when a value is not a list of lists."""

    def __init__(self, name: str, value: Any) -> None:
        super().__init__(
            f"Argument '{name}' must be a list of lists; got {value!r} of type {type(value).__name__}. "
            "Wrap items in a list."
        )


class ParameterError(ValidationError):
    """Raised when a parameter falls outside its allowed range."""

    def __init__(self, name: str, value: Any, constraint: str) -> None:
        super().__init__(
            f"Invalid value for '{name}': {value!r} (type {type(value).__name__}). {constraint}. "
            "Check and adjust this parameter."
        )


_ALLOWED_CENSORING = {"uniform", "exponential"}


def ensure_positive_int(value: int, name: str) -> None:
    """Ensure ``value`` is a positive integer."""
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise PositiveIntegerError(name, value)


def ensure_positive(value: float | int, name: str) -> None:
    """Ensure ``value`` is a positive number."""
    if not isinstance(value, Real) or isinstance(value, bool) or value <= 0:
        raise PositiveValueError(name, value)


def ensure_probability(value: float | int, name: str) -> None:
    """Ensure ``value`` lies in the closed interval [0, 1]."""
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not (0 <= float(value) <= 1)
    ):
        raise ParameterError(name, value, "must be between 0 and 1")


def ensure_in_choices(value: str, name: str, choices: Iterable[str]) -> None:
    """Ensure ``value`` is one of the allowed options.

    Parameters
    ----------
    value:
        Value provided by the user.
    name:
        Name of the argument being validated. Used in error messages.
    choices:
        Iterable of valid string options.

    Raises
    ------
    ChoiceError
        If ``value`` is not present in ``choices``.
    """
    if value not in choices:
        raise ChoiceError(name, value, choices)


def ensure_sequence_length(seq: Sequence[Any], length: int, name: str) -> None:
    """Ensure a sequence has an expected number of elements.

    Parameters
    ----------
    seq:
        Sequence-like object (e.g., ``list`` or ``tuple``).
    length:
        Required number of elements in ``seq``.
    name:
        Parameter name for error reporting.

    Raises
    ------
    LengthError
        If ``seq`` does not contain exactly ``length`` elements.
    """
    if len(seq) != length:
        raise LengthError(name, len(seq), length)


def _to_float_array(seq: Sequence[Any], name: str) -> NDArray[np.float64]:
    """Convert ``seq`` to a NumPy float64 array or raise an error."""
    try:
        arr = np.asarray(seq, dtype=float)
    except (TypeError, ValueError) as exc:
        for idx, val in enumerate(seq):
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, float)):
                raise NumericSequenceError(name, val, idx) from exc
        raise NumericSequenceError(name, seq) from exc

    for idx, val in enumerate(seq):
        if isinstance(val, (bool, np.bool_)):
            raise NumericSequenceError(name, val, idx)

    return arr


def ensure_numeric_sequence(seq: Sequence[Any], name: str) -> None:
    """Validate that a sequence consists solely of numbers.

    Parameters
    ----------
    seq:
        Sequence whose elements should all be ``int`` or ``float``.
    name:
        Parameter name for error reporting.

    Raises
    ------
    NumericSequenceError
        If any element cannot be interpreted as a numeric value.
    """
    _to_float_array(seq, name)


def ensure_positive_sequence(seq: Sequence[float], name: str) -> None:
    """Validate that a sequence contains only positive numbers.

    Parameters
    ----------
    seq:
        Sequence of numeric values.
    name:
        Parameter name for error reporting.

    Raises
    ------
    PositiveSequenceError
        If any element is less than or equal to zero. The offending value and
        its index are reported in the error message.
    """
    arr = _to_float_array(seq, name)
    nonpos = np.where((arr <= 0) | ~np.isfinite(arr))[0]
    if nonpos.size:
        idx = int(nonpos[0])
        raise PositiveSequenceError(name, seq[idx], idx)


def ensure_censoring_model(model_cens: str) -> None:
    """Validate that the censoring model is supported.

    Parameters
    ----------
    model_cens:
        Censoring model name provided by the user.

    Raises
    ------
    ChoiceError
        If ``model_cens`` is not one of ``"uniform"`` or ``"exponential"``.
    """
    ensure_in_choices(model_cens, "model_cens", _ALLOWED_CENSORING)


# Generator-specific validation helpers

_BETA_LEN = 3
_CMM_RATE_LEN = 6
_THMM_RATE_LEN = 3
_WEIBULL_DIST_PAR_LEN = 4
_EXP_DIST_PAR_LEN = 2


def _validate_base(n: int, model_cens: str, cens_par: float) -> None:
    """Common checks for sample size and censoring model."""
    ensure_positive_int(n, "n")
    ensure_censoring_model(model_cens)
    ensure_positive(cens_par, "cens_par")


def _validate_beta(beta: Sequence[float]) -> None:
    """Ensure beta is a numeric sequence of length three."""
    ensure_sequence_length(beta, _BETA_LEN, "beta")
    ensure_numeric_sequence(beta, "beta")


def _validate_aft_common(
    n: int, beta: Sequence[float], model_cens: str, cens_par: float
) -> None:
    """Shared validation logic for AFT generators."""
    _validate_base(n, model_cens, cens_par)
    ensure_numeric_sequence(beta, "beta")


def _validate_covariate_inputs(
    n: int,
    n_covariates: int | None,
    model_cens: str,
    cens_par: float,
    covariate_dist: str,
    max_time: float | None = None,
) -> None:
    """Common checks for generators with covariates.

    Parameters
    ----------
    n:
        Number of samples to generate.
    n_covariates:
        Expected number of covariates or ``None`` to skip the check.
    model_cens:
        Name of the censoring model.
    cens_par:
        Parameter for the censoring model.
    covariate_dist:
        Name of the covariate distribution.
    max_time:
        Optional maximum follow-up time. If provided, must be positive.
    """
    _validate_base(n, model_cens, cens_par)
    if n_covariates is not None:
        ensure_positive_int(n_covariates, "n_covariates")
    if max_time is not None:
        ensure_positive(max_time, "max_time")
    ensure_in_choices(covariate_dist, "covariate_dist", {"normal", "uniform", "binary"})


def validate_gen_cphm_inputs(
    n: int, model_cens: str, cens_par: float, covariate_range: float
) -> None:
    """Validate input parameters for CPHM data generation."""
    _validate_base(n, model_cens, cens_par)
    ensure_positive(covariate_range, "covariate_range")


def validate_gen_cmm_inputs(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
) -> None:
    """Validate inputs for generating CMM (Continuous-Time Markov Model) data."""
    _validate_base(n, model_cens, cens_par)
    _validate_beta(beta)
    ensure_positive(covariate_range, "covariate_range")
    ensure_sequence_length(rate, _CMM_RATE_LEN, "rate")


def validate_gen_tdcm_inputs(
    n: int,
    dist: str,
    corr: float,
    dist_par: Sequence[float],
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    lam: float,
) -> None:
    """Validate inputs for generating TDCM (Time-Dependent Covariate Model) data."""
    _validate_base(n, model_cens, cens_par)
    ensure_in_choices(dist, "dist", {"weibull", "exponential"})

    if dist == "weibull":
        if not (0 < corr <= 1):
            raise ParameterError("corr", corr, "with dist='weibull' must be in (0,1]")
        ensure_sequence_length(dist_par, _WEIBULL_DIST_PAR_LEN, "dist_par")
        ensure_positive_sequence(dist_par, "dist_par")

    if dist == "exponential":
        if not (-1 <= corr <= 1):
            raise ParameterError(
                "corr", corr, "with dist='exponential' must be in [-1,1]"
            )
        ensure_sequence_length(dist_par, _EXP_DIST_PAR_LEN, "dist_par")
        ensure_positive_sequence(dist_par, "dist_par")

    _validate_beta(beta)
    ensure_positive(lam, "lambda")


def validate_gen_thmm_inputs(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
) -> None:
    """Validate inputs for generating THMM (Time-Homogeneous Markov Model) data."""
    _validate_base(n, model_cens, cens_par)
    _validate_beta(beta)
    ensure_positive(covariate_range, "covariate_range")
    ensure_sequence_length(rate, _THMM_RATE_LEN, "rate")


def validate_dg_biv_inputs(
    n: int, dist: str, corr: float, dist_par: Sequence[float]
) -> None:
    """Validate inputs for the :func:`sample_bivariate_distribution` helper."""
    ensure_positive_int(n, "n")
    ensure_in_choices(dist, "dist", {"weibull", "exponential"})

    if not isinstance(corr, (int, float)) or not (-1 < corr < 1):
        raise ParameterError("corr", corr, "must be a numeric value between -1 and 1")

    ensure_positive_sequence(dist_par, "dist_par")
    if dist == "exponential":
        ensure_sequence_length(dist_par, _EXP_DIST_PAR_LEN, "dist_par")
    if dist == "weibull":
        ensure_sequence_length(dist_par, _WEIBULL_DIST_PAR_LEN, "dist_par")


def validate_gen_aft_log_normal_inputs(
    n: int,
    beta: Sequence[float],
    sigma: float,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for the log-normal AFT generator."""
    _validate_aft_common(n, beta, model_cens, cens_par)
    ensure_positive(sigma, "sigma")


def validate_gen_aft_weibull_inputs(
    n: int,
    beta: Sequence[float],
    shape: float,
    scale: float,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for the Weibull AFT generator."""
    _validate_aft_common(n, beta, model_cens, cens_par)
    ensure_positive(shape, "shape")
    ensure_positive(scale, "scale")


def validate_gen_aft_log_logistic_inputs(
    n: int,
    beta: Sequence[float],
    shape: float,
    scale: float,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for the log-logistic AFT generator."""
    _validate_aft_common(n, beta, model_cens, cens_par)
    ensure_positive(shape, "shape")
    ensure_positive(scale, "scale")


def validate_competing_risks_inputs(
    n: int,
    n_risks: int,
    baseline_hazards: Sequence[float] | None,
    betas: Sequence[Sequence[float]] | None,
    covariate_dist: str,
    max_time: float | None,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for competing risks data generation."""
    _validate_covariate_inputs(n, None, model_cens, cens_par, covariate_dist, max_time)
    ensure_positive_int(n_risks, "n_risks")

    if baseline_hazards is not None:
        ensure_sequence_length(baseline_hazards, n_risks, "baseline_hazards")
        ensure_positive_sequence(baseline_hazards, "baseline_hazards")

    if betas is not None:
        if not isinstance(betas, list) or any(not isinstance(b, list) for b in betas):
            raise ListOfListsError("betas", betas)
        for b in betas:
            ensure_numeric_sequence(b, "betas")


def validate_piecewise_params(
    breakpoints: Sequence[float], hazard_rates: Sequence[float]
) -> None:
    """Validate breakpoint and hazard rate sequences."""
    ensure_sequence_length(hazard_rates, len(breakpoints) + 1, "hazard_rates")
    ensure_positive_sequence(breakpoints, "breakpoints")
    ensure_positive_sequence(hazard_rates, "hazard_rates")
    if np.any(np.diff(breakpoints) <= 0):
        raise ParameterError(
            "breakpoints",
            breakpoints,
            "must be a strictly increasing sequence. Sort the list and remove duplicates.",
        )


def validate_gen_piecewise_inputs(
    n: int,
    breakpoints: Sequence[float],
    hazard_rates: Sequence[float],
    n_covariates: int,
    model_cens: str,
    cens_par: float,
    covariate_dist: str,
) -> None:
    """Validate parameters for :func:`gen_piecewise_exponential`."""
    _validate_covariate_inputs(n, n_covariates, model_cens, cens_par, covariate_dist)
    validate_piecewise_params(breakpoints, hazard_rates)


def validate_gen_mixture_inputs(
    n: int,
    cure_fraction: float,
    baseline_hazard: float,
    n_covariates: int,
    model_cens: str,
    cens_par: float,
    max_time: float | None,
    covariate_dist: str,
) -> None:
    """Validate parameters for :func:`gen_mixture_cure`."""
    _validate_covariate_inputs(
        n, n_covariates, model_cens, cens_par, covariate_dist, max_time
    )
    ensure_positive(baseline_hazard, "baseline_hazard")
    if not 0 < cure_fraction < 1:
        raise ParameterError(
            "cure_fraction",
            cure_fraction,
            "must be between 0 and 1 (exclusive). Try a value like 0.5",
        )
