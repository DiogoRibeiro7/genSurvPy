import pandas as pd
import pytest

from gen_surv.piecewise import gen_piecewise_exponential


def test_gen_piecewise_exponential_runs():
    df = gen_piecewise_exponential(
        n=10, breakpoints=[1.0], hazard_rates=[0.5, 1.0], seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert {"time", "status"}.issubset(df.columns)


def test_piecewise_invalid_lengths():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5, breakpoints=[1.0, 2.0], hazard_rates=[0.5], seed=42
        )


def test_piecewise_invalid_hazard_and_breakpoints():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[2.0, 1.0],
            hazard_rates=[0.5, 1.0, 1.5],
            seed=42,
        )
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[1.0],
            hazard_rates=[0.5, -1.0],
            seed=42,
        )


def test_piecewise_covariate_distributions():
    for dist, params in [
        ("uniform", {"low": 0.0, "high": 1.0}),
        ("binary", {"p": 0.7}),
    ]:
        df = gen_piecewise_exponential(
            n=5,
            breakpoints=[1.0],
            hazard_rates=[0.2, 0.4],
            covariate_dist=dist,
            covariate_params=params,
            seed=1,
        )
        assert len(df) == 5
        assert {"X0", "X1"}.issubset(df.columns)


def test_piecewise_custom_betas_reproducible():
    df1 = gen_piecewise_exponential(
        n=5,
        breakpoints=[1.0],
        hazard_rates=[0.1, 0.2],
        betas=[0.5, -0.2],
        seed=2,
    )
    df2 = gen_piecewise_exponential(
        n=5,
        breakpoints=[1.0],
        hazard_rates=[0.1, 0.2],
        betas=[0.5, -0.2],
        seed=2,
    )
    pd.testing.assert_frame_equal(df1, df2)


def test_piecewise_invalid_covariate_dist():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[1.0],
            hazard_rates=[0.5, 1.0],
            covariate_dist="unknown",
            seed=1,
        )


def test_piecewise_invalid_censoring_model():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[1.0],
            hazard_rates=[0.5, 1.0],
            model_cens="bad",
            seed=1,
        )


def test_piecewise_negative_breakpoint():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[-1.0],
            hazard_rates=[0.5, 1.0],
            seed=1,
        )
