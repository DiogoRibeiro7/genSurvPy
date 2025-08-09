import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.tdcm import gen_tdcm


def test_tdcm_generation_benchmark(benchmark):
    benchmark(
        gen_tdcm,
        n=1000,
        dist="weibull",
        corr=0.5,
        dist_par=[1, 2, 1, 2],
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        lam=1.0,
    )
