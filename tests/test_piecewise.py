import pandas as pd
import pytest
from gen_surv.piecewise import gen_piecewise_exponential


def test_gen_piecewise_exponential_runs():
    df = gen_piecewise_exponential(
        n=10,
        breakpoints=[1.0],
        hazard_rates=[0.5, 1.0],
        seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10
    assert {"time", "status"}.issubset(df.columns)


def test_piecewise_invalid_lengths():
    with pytest.raises(ValueError):
        gen_piecewise_exponential(
            n=5,
            breakpoints=[1.0, 2.0],
            hazard_rates=[0.5],
            seed=42
        )
