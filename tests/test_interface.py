import pytest

from gen_surv import generate
from gen_surv.validation import ValidationError


def test_generate_tdcm_runs():
    df = generate(
        model="tdcm",
        n=10,
        dist="weibull",
        corr=0.5,
        dist_par=[1, 2, 1, 2],
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        lam=1.0,
    )
    assert not df.empty


def test_generate_invalid_model():
    with pytest.raises(ValueError):
        generate(model="unknown")


def test_generate_error_message_includes_model():
    with pytest.raises(ValidationError) as exc:
        generate(
            model="cphm",
            n=0,
            model_cens="uniform",
            cens_par=1.0,
            beta=0.5,
            covariate_range=2.0,
        )
    assert "model 'cphm'" in str(exc.value)
