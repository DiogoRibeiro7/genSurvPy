from __future__ import annotations

from gen_surv import generate
from tests.conftest import assert_frame_numeric_equal


def test_generate_reproducible_with_seed() -> None:
    cfg = dict(
        model="cphm",
        n=32,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=0.7,
        seed=1234,
    )
    df1 = generate(**cfg)
    df2 = generate(**cfg)
    assert_frame_numeric_equal(df1, df2)
