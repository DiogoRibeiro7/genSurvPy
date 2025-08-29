import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.cmm import gen_cmm


def test_cmm_generation_benchmark(benchmark):
    benchmark(
        gen_cmm,
        n=1000,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        seed=42,
    )
