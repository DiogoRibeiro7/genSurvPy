import pytest

pytest.importorskip("pytest_benchmark")

from gen_surv.mixture import gen_mixture_cure


def test_mixture_generation_benchmark(benchmark):
    benchmark(
        gen_mixture_cure,
        n=1000,
        cure_fraction=0.3,
        seed=42,
    )
