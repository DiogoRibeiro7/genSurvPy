import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.cphm import gen_cphm


def test_cphm_generation_benchmark(benchmark):
    benchmark(
        gen_cphm,
        n=1000,
        model_cens="uniform",
        cens_par=1.0,
        beta=0.5,
        covariate_range=2.0,
        seed=42,
    )
