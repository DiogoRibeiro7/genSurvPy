"""Input validation utilities.

This module unifies the low-level validation helpers and the higher-level
checks used by the data generators.
"""

from __future__ import annotations

import math
import warnings
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


def ensure_finite(value: float | int, name: str) -> None:
    """Ensure ``value`` is a real number that is neither NaN nor infinite.

    Every comparison with NaN is false, so a check written as ``value <= 0``
    silently admits it, and ``inf > 0`` is true. Both then reach NumPy, where
    they either surface as an unrelated error -- ``OverflowError: high - low
    range exceeds valid bounds`` from a uniform draw -- or produce a frame
    quietly full of NaN. Rejecting them here is what makes the message name the
    argument the caller got wrong.
    """
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ParameterError(name, value, "must be a number")
    if not math.isfinite(float(value)):
        raise ParameterError(name, value, "must be a finite number")


def ensure_positive(value: float | int, name: str) -> None:
    """Ensure ``value`` is a finite positive number."""
    if not isinstance(value, Real) or isinstance(value, bool):
        raise PositiveValueError(name, value)
    if not math.isfinite(float(value)):
        # A separate message: NaN and infinity are numbers, and saying so is
        # more use than "must be greater than 0" for a value no comparison
        # would have caught.
        raise ParameterError(name, value, "must be a finite number")
    if value <= 0:
        raise PositiveValueError(name, value)


def ensure_probability(value: float | int, name: str) -> None:
    """Ensure ``value`` lies in the closed interval [0, 1]."""
    ensure_finite(value, name)
    if not (0 <= float(value) <= 1):
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
    arr = _to_float_array(seq, name)
    bad = np.where(~np.isfinite(arr))[0]
    if bad.size:
        idx = int(bad[0])
        raise ParameterError(f"{name}[{idx}]", seq[idx], "must be a finite number")


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
_TDCM_BETA_LEN = 2
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


def _validate_tdcm_beta(beta: Sequence[float]) -> None:
    """Ensure beta matches the two coefficients the TDCM actually uses.

    The model has one coefficient for the baseline covariate and one for the
    effect of the time-dependent covariate. Releases up to 1.2.0 required three
    and silently ignored the third, so a length of three is still accepted with
    a deprecation warning to avoid breaking existing callers.
    """
    ensure_numeric_sequence(beta, "beta")

    if len(beta) == _TDCM_BETA_LEN:
        return

    if len(beta) == _BETA_LEN:
        warnings.warn(
            "gen_tdcm uses two coefficients; passing three is deprecated because "
            "the third is ignored, and it will raise in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        return

    raise LengthError("beta", len(beta), _TDCM_BETA_LEN)


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
    n: int,
    model_cens: str,
    cens_par: float,
    covariate_range: float,
    beta: float | None = None,
) -> None:
    """Validate input parameters for CPHM data generation.

    ``beta`` is a log hazard ratio and may be any sign, so no positivity check
    reaches it. It still has to be a finite number: NaN propagates into every
    drawn time, and the frame comes back the right shape and entirely NaN.
    """
    _validate_base(n, model_cens, cens_par)
    ensure_positive(covariate_range, "covariate_range")
    if beta is not None:
        ensure_finite(beta, "beta")


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
    # Only the length was checked. A negative entry reached NumPy and surfaced
    # as "ValueError: scale < 0" from inside the generator; NaN passed straight
    # through into every drawn time.
    ensure_positive_sequence(rate, "rate")


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

    _validate_tdcm_beta(beta)
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
    ensure_positive_sequence(rate, "rate")


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


_RECURRENT_PROCESSES = {"ag", "pwp_tt", "pwp_gt"}
_BASELINE_KEYS: dict[str, set[str]] = {
    "exponential": {"rate"},
    "weibull": {"shape", "scale"},
    "gompertz": {"rate", "shape"},
}


def _validate_baseline_params(
    baseline: object, baseline_params: dict[str, float] | None
) -> None:
    """Check the parameters supplied for a baseline hazard family.

    Unknown keys are rejected rather than ignored, so a misspelt parameter
    surfaces immediately instead of silently leaving the default in place.

    ``baseline`` may also be an object implementing the ``BaselineHazard``
    protocol, which has already validated itself on construction. Supplying
    parameters alongside such an object is ambiguous -- they could not be
    applied without rebuilding it -- so that combination is rejected.
    """
    if not isinstance(baseline, str):
        required = ("cumulative_hazard", "inverse_cumulative_hazard")
        if not all(callable(getattr(baseline, name, None)) for name in required):
            raise ParameterError(
                "baseline",
                baseline,
                "must be a baseline name or an object implementing "
                "BaselineHazard, with cumulative_hazard and "
                "inverse_cumulative_hazard methods",
            )
        if baseline_params is not None:
            raise ParameterError(
                "baseline_params",
                baseline_params,
                "cannot be combined with a BaselineHazard object; set the "
                "parameters on the object itself",
            )
        return

    ensure_in_choices(baseline, "baseline", _BASELINE_KEYS.keys())

    if baseline_params is None:
        return

    if not isinstance(baseline_params, dict):
        raise ParameterError(
            "baseline_params", baseline_params, "must be a dict or None"
        )

    allowed = _BASELINE_KEYS[baseline]
    unknown = set(baseline_params) - allowed
    if unknown:
        raise ParameterError(
            "baseline_params",
            baseline_params,
            f"has no key(s) {sorted(unknown)} for baseline '{baseline}'; "
            f"allowed keys are {sorted(allowed)}",
        )

    for key, value in baseline_params.items():
        # A Gompertz shape may be negative, which is what makes the hazard
        # decline; every other parameter must be strictly positive.
        if baseline == "gompertz" and key == "shape":
            if not isinstance(value, Real) or isinstance(value, bool) or value == 0:
                raise ParameterError(
                    f"baseline_params['{key}']",
                    value,
                    "must be a non-zero number; a negative value gives a "
                    "declining hazard",
                )
            continue
        ensure_positive(value, f"baseline_params['{key}']")


def validate_gen_recurrent_events_inputs(
    n: int,
    process: str,
    baseline: object,
    baseline_params: dict[str, float] | None,
    n_covariates: int,
    stratum_effects: Sequence[float] | None,
    max_events: int | None,
    followup_time: float,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for :func:`gen_surv.recurrent.gen_recurrent_events`."""
    ensure_positive_int(n, "n")
    ensure_positive_int(n_covariates, "n_covariates")
    ensure_censoring_model(model_cens)
    ensure_positive(cens_par, "cens_par")
    ensure_positive(followup_time, "followup_time")
    ensure_in_choices(process, "process", _RECURRENT_PROCESSES)
    _validate_baseline_params(baseline, baseline_params)

    if stratum_effects is not None:
        # Andersen-Gill is defined by an intensity that does not depend on the
        # event history, so per-event effects contradict the process. Silently
        # applying them would mislabel PWP data as AG, and silently dropping
        # them would discard an argument the caller clearly meant.
        if process == "ag":
            raise ParameterError(
                "stratum_effects",
                stratum_effects,
                "is not applicable to process='ag', whose intensity cannot "
                "depend on the event number; use process='pwp_tt' or "
                "process='pwp_gt'",
            )
        ensure_numeric_sequence(stratum_effects, "stratum_effects")
        ensure_positive_sequence(stratum_effects, "stratum_effects")
        if len(stratum_effects) == 0:
            raise ParameterError(
                "stratum_effects", stratum_effects, "must not be empty"
            )

    if max_events is not None:
        ensure_positive_int(max_events, "max_events")
