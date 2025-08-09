import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.piecewise import gen_piecewise_exponential


def test_piecewise_generation_benchmark(benchmark):
    benchmark(
        gen_piecewise_exponential,
        n=1000,
        breakpoints=[1.0, 2.0],
        hazard_rates=[0.5, 0.3, 0.1],
        n_covariates=3,
        seed=42,
    )
