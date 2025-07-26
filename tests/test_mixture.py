import pandas as pd
from gen_surv.mixture import gen_mixture_cure, cure_fraction_estimate


def test_gen_mixture_cure_runs():
    df = gen_mixture_cure(n=10, cure_fraction=0.3, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert {"time", "status", "cured"}.issubset(df.columns)


def test_cure_fraction_estimate_range():
    df = gen_mixture_cure(n=50, cure_fraction=0.3, seed=0)
    est = cure_fraction_estimate(df)
    assert 0 <= est <= 1
