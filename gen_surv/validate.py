"""Validation utilities for data generators."""

from __future__ import annotations

from collections.abc import Sequence

from ._validation import (
    ensure_censoring_model,
    ensure_in_choices,
    ensure_numeric_sequence,
    ensure_positive,
    ensure_positive_int,
    ensure_positive_sequence,
    ensure_sequence_length,
    ListOfListsError,
    ParameterError,
)


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
    """Validate inputs for generating CMM (Continuous-Time Markov Model) data.

    Parameters
    ----------
    n : int
        Sample size.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    beta : Sequence[float]
        Regression coefficients.
    covariate_range : float
        Range of the uniform covariate distribution.
    rate : Sequence[float]
        Six transition rate parameters.
    """

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
    """Validate inputs for generating TDCM (Time-Dependent Covariate Model) data.

    Parameters
    ----------
    n : int
        Sample size.
    dist : {"weibull", "exponential"}
        Distribution used to generate correlated covariates.
    corr : float
        Correlation coefficient for the bivariate distribution.
    dist_par : Sequence[float]
        Parameters of the chosen distribution.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    beta : Sequence[float]
        Regression coefficients.
    lam : float
        Baseline hazard rate.
    """

    _validate_base(n, model_cens, cens_par)
    ensure_in_choices(dist, "dist", {"weibull", "exponential"})

    if dist == "weibull":
        if not (0 < corr <= 1):
            raise ParameterError("corr", corr, "with dist='weibull' must be in (0,1]")
        ensure_sequence_length(dist_par, _WEIBULL_DIST_PAR_LEN, "dist_par")
        ensure_positive_sequence(dist_par, "dist_par")

    if dist == "exponential":
        if not (-1 <= corr <= 1):
            raise ParameterError("corr", corr, "with dist='exponential' must be in [-1,1]")
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
    """Validate inputs for generating THMM (Time-Homogeneous Markov Model) data.

    Parameters
    ----------
    n : int
        Sample size.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    beta : Sequence[float]
        Regression coefficients.
    covariate_range : float
        Range of the uniform covariate distribution.
    rate : Sequence[float]
        Three transition rate parameters.
    """

    _validate_base(n, model_cens, cens_par)
    _validate_beta(beta)
    ensure_positive(covariate_range, "covariate_range")
    ensure_sequence_length(rate, _THMM_RATE_LEN, "rate")


def validate_dg_biv_inputs(
    n: int, dist: str, corr: float, dist_par: Sequence[float]
) -> None:
    """Validate inputs for the :func:`sample_bivariate_distribution` helper.

    Parameters
    ----------
    n : int
        Number of samples.
    dist : {"weibull", "exponential"}
        Bivariate marginal distribution.
    corr : float
        Correlation coefficient.
    dist_par : Sequence[float]
        Parameters for the selected distribution.
    """

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
    """Validate parameters for the log-normal AFT generator.

    Parameters
    ----------
    n : int
        Sample size.
    beta : Sequence[float]
        Regression coefficients.
    sigma : float
        Scale parameter of the log-normal distribution.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    """

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
    """Validate parameters for the Weibull AFT generator.

    Parameters
    ----------
    n : int
        Sample size.
    beta : Sequence[float]
        Regression coefficients.
    shape : float
        Shape parameter of the Weibull distribution.
    scale : float
        Scale parameter of the Weibull distribution.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    """

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
    """Validate parameters for the log-logistic AFT generator.

    Parameters
    ----------
    n : int
        Sample size.
    beta : Sequence[float]
        Regression coefficients.
    shape : float
        Shape parameter of the log-logistic distribution.
    scale : float
        Scale parameter of the log-logistic distribution.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    """

    _validate_aft_common(n, beta, model_cens, cens_par)
    ensure_positive(shape, "shape")
    ensure_positive(scale, "scale")


def validate_competing_risks_inputs(
    n: int,
    n_risks: int,
    baseline_hazards: Sequence[float] | None,
    betas: Sequence[Sequence[float]] | None,
    model_cens: str,
    cens_par: float,
) -> None:
    """Validate parameters for competing risks data generation.

    Parameters
    ----------
    n : int
        Sample size.
    n_risks : int
        Number of competing risks.
    baseline_hazards : Sequence[float] or None
        Baseline hazard for each risk.
    betas : Sequence[Sequence[float]] or None
        Regression coefficients for each risk.
    model_cens : str
        Censoring model identifier.
    cens_par : float
        Censoring distribution parameter.
    """

    _validate_base(n, model_cens, cens_par)
    ensure_positive_int(n_risks, "n_risks")

    if baseline_hazards is not None:
        ensure_sequence_length(baseline_hazards, n_risks, "baseline_hazards")
        ensure_positive_sequence(baseline_hazards, "baseline_hazards")

    if betas is not None:
        if not isinstance(betas, list) or any(not isinstance(b, list) for b in betas):
            raise ListOfListsError("betas", betas)
        for b in betas:
            ensure_numeric_sequence(b, "betas")

