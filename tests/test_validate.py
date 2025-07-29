import pytest

import gen_surv.validate as v


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


def test_validate_gen_aft_log_normal_inputs_valid():
    """Valid parameters should not raise an error for AFT log-normal."""
    v.validate_gen_aft_log_normal_inputs(
        1,
        [0.1, 0.2],
        1.0,
        "uniform",
        0.5,
    )


def test_validate_dg_biv_inputs_valid_weibull():
    """Valid parameters for a Weibull distribution should pass."""
    v.validate_dg_biv_inputs(5, "weibull", 0.1, [1.0, 1.0, 1.0, 1.0])
