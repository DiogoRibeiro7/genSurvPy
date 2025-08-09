from __future__ import annotations

import pytest

from gen_surv import generate


@pytest.mark.slow
def test_generate_perf_smoke(benchmark) -> None:
    def _run():
        return generate(
            model="cphm",
            n=50_000,
            beta=0.5,
            covariate_range=2.0,
            model_cens="uniform",
            cens_par=0.7,
            seed=123,
        )

    df = benchmark(_run)
    assert len(df) == 50_000
