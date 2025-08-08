import numpy as np
import pytest

import gen_surv.validation as v
from gen_surv.validation import (
    ChoiceError,
    ParameterError,
    PositiveIntegerError,
    PositiveValueError,
    ensure_censoring_model,
    ensure_positive,
    ensure_positive_int,
)


def test_validate_gen_cphm_inputs_valid():
    """Ensure valid inputs pass without raising an exception."""
    v.validate_gen_cphm_inputs(1, "uniform", 0.5, 1.0)


@pytest.mark.parametrize(
    "n, model_cens, cens_par, covariate_range",
    [
        (0, "uniform", 0.5, 1.0),
        (1, "bad", 0.5, 1.0),
        (1, "uniform", -1.0, 1.0),
        (1, "uniform", 0.5, -1.0),
    ],
)
def test_validate_gen_cphm_inputs_invalid(n, model_cens, cens_par, covariate_range):
    """Invalid parameter combinations should raise ValueError."""
    with pytest.raises(ValueError):
        v.validate_gen_cphm_inputs(n, model_cens, cens_par, covariate_range)


def test_validate_dg_biv_inputs_invalid():
    """Invalid distribution names should raise an error."""
    with pytest.raises(ValueError):
        v.validate_dg_biv_inputs(10, "normal", 0.1, [1, 1])


def test_validate_gen_cmm_inputs_invalid_beta_length():
    """Invalid beta length should raise a ValueError."""
    with pytest.raises(ValueError):
        v.validate_gen_cmm_inputs(
            1,
            "uniform",
            0.5,
            [0.1, 0.2],
            covariate_range=1.0,
            rate=[0.1] * 6,
        )


@pytest.mark.parametrize(
    "n, model_cens, cens_par, cov_range, rate",
    [
        (0, "uniform", 0.5, 1.0, [0.1] * 6),
        (1, "bad", 0.5, 1.0, [0.1] * 6),
        (1, "uniform", 0.0, 1.0, [0.1] * 6),
        (1, "uniform", 0.5, 0.0, [0.1] * 6),
        (1, "uniform", 0.5, 1.0, [0.1] * 3),
    ],
)
def test_validate_gen_cmm_inputs_other_invalid(
    n, model_cens, cens_par, cov_range, rate
):
    with pytest.raises(ValueError):
        v.validate_gen_cmm_inputs(
            n, model_cens, cens_par, [0.1, 0.2, 0.3], cov_range, rate
        )


def test_validate_gen_cmm_inputs_valid():
    v.validate_gen_cmm_inputs(
        1, "uniform", 1.0, [0.1, 0.2, 0.3], covariate_range=1.0, rate=[0.1] * 6
    )


def test_validate_gen_tdcm_inputs_invalid_lambda():
    """Lambda <= 0 should raise a ValueError."""
    with pytest.raises(ValueError):
        v.validate_gen_tdcm_inputs(
            1,
            "weibull",
            0.5,
            [1, 2, 1, 2],
            "uniform",
            1.0,
            beta=[0.1, 0.2, 0.3],
            lam=0,
        )


@pytest.mark.parametrize(
    "dist,corr,dist_par",
    [
        ("bad", 0.5, [1, 2]),
        ("weibull", 0.0, [1, 2, 3, 4]),
        ("weibull", 0.5, [1, 2, -1, 2]),
        ("weibull", 0.5, [1, 2, 3]),
        ("exponential", 2.0, [1, 1]),
        ("exponential", 0.5, [1]),
    ],
)
def test_validate_gen_tdcm_inputs_invalid_dist(dist, corr, dist_par):
    with pytest.raises(ValueError):
        v.validate_gen_tdcm_inputs(
            1,
            dist,
            corr,
            dist_par,
            "uniform",
            1.0,
            beta=[0.1, 0.2, 0.3],
            lam=1.0,
        )


def test_validate_gen_tdcm_inputs_valid():
    v.validate_gen_tdcm_inputs(
        1,
        "weibull",
        0.5,
        [1, 1, 1, 1],
        "uniform",
        1.0,
        beta=[0.1, 0.2, 0.3],
        lam=1.0,
    )


def test_validate_gen_aft_log_normal_inputs_valid():
    """Valid parameters should not raise an error for AFT log-normal."""
    v.validate_gen_aft_log_normal_inputs(
        1,
        [0.1, 0.2],
        1.0,
        "uniform",
        0.5,
    )


@pytest.mark.parametrize(
    "n,beta,sigma,model_cens,cens_par",
    [
        (0, [0.1], 1.0, "uniform", 1.0),
        (1, "bad", 1.0, "uniform", 1.0),
        (1, [0.1], 0.0, "uniform", 1.0),
        (1, [0.1], 1.0, "bad", 1.0),
        (1, [0.1], 1.0, "uniform", 0.0),
    ],
)
def test_validate_gen_aft_log_normal_inputs_invalid(
    n, beta, sigma, model_cens, cens_par
):
    with pytest.raises(ValueError):
        v.validate_gen_aft_log_normal_inputs(n, beta, sigma, model_cens, cens_par)


def test_validate_dg_biv_inputs_valid_weibull():
    """Valid parameters for a Weibull distribution should pass."""
    v.validate_dg_biv_inputs(5, "weibull", 0.1, [1.0, 1.0, 1.0, 1.0])


def test_validate_dg_biv_inputs_invalid_corr_and_params():
    with pytest.raises(ValueError):
        v.validate_dg_biv_inputs(1, "exponential", -2.0, [1.0, 1.0])
    with pytest.raises(ValueError):
        v.validate_dg_biv_inputs(1, "exponential", 0.5, [1.0])
    with pytest.raises(ValueError):
        v.validate_dg_biv_inputs(1, "weibull", 0.5, [1.0, 1.0])


def test_validate_gen_aft_weibull_inputs_and_log_logistic():
    with pytest.raises(ValueError):
        v.validate_gen_aft_weibull_inputs(0, [0.1], 1.0, 1.0, "uniform", 1.0)
    with pytest.raises(ValueError):
        v.validate_gen_aft_log_logistic_inputs(1, [0.1], -1.0, 1.0, "uniform", 1.0)


@pytest.mark.parametrize(
    "shape,scale",
    [(-1.0, 1.0), (1.0, -1.0)],
)
def test_validate_gen_aft_weibull_invalid_params(shape, scale):
    with pytest.raises(ValueError):
        v.validate_gen_aft_weibull_inputs(1, [0.1], shape, scale, "uniform", 1.0)


def test_validate_gen_aft_weibull_valid():
    v.validate_gen_aft_weibull_inputs(1, [0.1], 1.0, 1.0, "uniform", 1.0)


def test_validate_gen_aft_log_logistic_valid():
    v.validate_gen_aft_log_logistic_inputs(1, [0.1], 1.0, 1.0, "uniform", 1.0)


def test_positive_sequence_nan_inf():
    with pytest.raises(v.PositiveSequenceError):
        v.ensure_positive_sequence([1.0, float("nan")], "x")
    with pytest.raises(v.PositiveSequenceError):
        v.ensure_positive_sequence([1.0, float("inf")], "x")


def test_numeric_sequence_rejects_bool():
    with pytest.raises(v.NumericSequenceError):
        v.ensure_numeric_sequence([1, True], "x")


def test_validate_competing_risks_inputs():
    with pytest.raises(ValueError):
        v.validate_competing_risks_inputs(1, 2, [0.1], None, "uniform", 1.0)
    v.validate_competing_risks_inputs(1, 1, [0.5], [[0.1]], "uniform", 0.5)


@pytest.mark.parametrize(
    "n,model_cens,cens_par,beta,cov_range,rate",
    [
        (0, "uniform", 1.0, [0.1, 0.2, 0.3], 1.0, [0.1, 0.2, 0.3]),
        (1, "bad", 1.0, [0.1, 0.2, 0.3], 1.0, [0.1, 0.2, 0.3]),
        (1, "uniform", 0.0, [0.1, 0.2, 0.3], 1.0, [0.1, 0.2, 0.3]),
        (1, "uniform", 1.0, [0.1, 0.2], 1.0, [0.1, 0.2, 0.3]),
        (1, "uniform", 1.0, [0.1, 0.2, 0.3], 0.0, [0.1, 0.2, 0.3]),
        (1, "uniform", 1.0, [0.1, 0.2, 0.3], 1.0, [0.1]),
    ],
)
def test_validate_gen_thmm_inputs_invalid(
    n, model_cens, cens_par, beta, cov_range, rate
):
    with pytest.raises(ValueError):
        v.validate_gen_thmm_inputs(n, model_cens, cens_par, beta, cov_range, rate)


def test_validate_gen_thmm_inputs_valid():
    v.validate_gen_thmm_inputs(1, "uniform", 1.0, [0.1, 0.2, 0.3], 1.0, [0.1, 0.2, 0.3])


def test_positive_integer_error():
    with pytest.raises(PositiveIntegerError):
        ensure_positive_int(-1, "n")


def test_ensure_positive_int_accepts_numpy_and_rejects_bool():
    ensure_positive_int(np.int64(5), "n")
    with pytest.raises(PositiveIntegerError):
        ensure_positive_int(True, "n")


def test_ensure_positive_accepts_numpy_and_rejects_bool():
    ensure_positive(np.float64(0.1), "val")
    with pytest.raises(PositiveValueError):
        ensure_positive(True, "val")


def test_censoring_model_choice_error():
    with pytest.raises(ChoiceError):
        ensure_censoring_model("bad")


def test_parameter_error_from_validator():
    with pytest.raises(ParameterError):
        v.validate_gen_tdcm_inputs(
            1,
            "weibull",
            0.0,
            [1, 2, 3, 4],
            "uniform",
            1.0,
            beta=[0.1, 0.2, 0.3],
            lam=1.0,
        )
