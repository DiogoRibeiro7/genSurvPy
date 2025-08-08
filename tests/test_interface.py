from gen_surv import generate
import pytest


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
