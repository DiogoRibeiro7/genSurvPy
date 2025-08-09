import pandas as pd
import pytest

from gen_surv.mixture import cure_fraction_estimate, gen_mixture_cure


def test_gen_mixture_cure_runs():
    df = gen_mixture_cure(n=10, cure_fraction=0.3, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert {"time", "status", "cured"}.issubset(df.columns)


def test_cure_fraction_estimate_range():
    df = gen_mixture_cure(n=50, cure_fraction=0.3, seed=0)
    est = cure_fraction_estimate(df)
    assert 0 <= est <= 1


def test_cure_fraction_estimate_empty_returns_zero():
    df = pd.DataFrame(columns=["time", "status"])
    est = cure_fraction_estimate(df)
    assert est == 0.0


def test_gen_mixture_cure_invalid_inputs():
    with pytest.raises(ValueError):
        gen_mixture_cure(n=5, cure_fraction=1.5)
    with pytest.raises(ValueError):
        gen_mixture_cure(n=5, cure_fraction=0.2, baseline_hazard=0)
    with pytest.raises(ValueError):
        gen_mixture_cure(n=5, cure_fraction=0.2, covariate_dist="bad")
    with pytest.raises(ValueError):
        gen_mixture_cure(n=5, cure_fraction=0.2, model_cens="bad")
    with pytest.raises(ValueError):
        gen_mixture_cure(
            n=5,
            cure_fraction=0.2,
            betas_survival=[0.1, 0.2],
            betas_cure=[0.1],
        )


def test_gen_mixture_cure_max_time_cap():
    df = gen_mixture_cure(
        n=50,
        cure_fraction=0.3,
        max_time=5.0,
        model_cens="exponential",
        cens_par=1.0,
        seed=123,
    )
    assert (df["time"] <= 5.0).all()


def test_cure_fraction_estimate_close_to_true():
    df = gen_mixture_cure(n=200, cure_fraction=0.4, seed=1)
    est = cure_fraction_estimate(df)
    assert pytest.approx(0.4, abs=0.15) == est


def test_gen_mixture_cure_covariate_distributions():
    for dist in ["uniform", "binary"]:
        df = gen_mixture_cure(n=20, cure_fraction=0.3, covariate_dist=dist, seed=2)
        assert {"time", "status", "cured", "X0", "X1"}.issubset(df.columns)


def test_gen_mixture_cure_no_max_time_allows_long_times():
    df = gen_mixture_cure(
        n=100,
        cure_fraction=0.5,
        max_time=None,
        model_cens="uniform",
        cens_par=20.0,
        seed=3,
    )
    assert df["time"].max() > 10


def test_cure_fraction_estimate_small_sample():
    df = gen_mixture_cure(n=3, cure_fraction=0.2, seed=4)
    est = cure_fraction_estimate(df)
    assert 0 <= est <= 1
