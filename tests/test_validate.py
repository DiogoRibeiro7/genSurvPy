import pytest
import gen_surv.validate as v


def test_validate_gen_cphm_inputs_valid():
    """Ensure valid inputs pass without raising an exception."""
    v.validate_gen_cphm_inputs(1, "uniform", 0.5, 1.0)


@pytest.mark.parametrize(
    "n, model_cens, cens_par, covar",
    [
        (0, "uniform", 0.5, 1.0),
        (1, "bad", 0.5, 1.0),
        (1, "uniform", -1.0, 1.0),
        (1, "uniform", 0.5, -1.0),
    ],
)
def test_validate_gen_cphm_inputs_invalid(n, model_cens, cens_par, covar):
    """Invalid parameter combinations should raise ValueError."""
    with pytest.raises(ValueError):
        v.validate_gen_cphm_inputs(n, model_cens, cens_par, covar)


def test_validate_dg_biv_inputs_invalid():
    """Invalid distribution names should raise an error."""
    with pytest.raises(ValueError):
        v.validate_dg_biv_inputs(10, "normal", 0.1, [1, 1])
