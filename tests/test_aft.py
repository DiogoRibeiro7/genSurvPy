import pandas as pd
from gen_surv.aft import gen_aft_log_normal

def test_gen_aft_log_normal_runs():
    df = gen_aft_log_normal(
        n=10,
        beta=[0.5, -0.2],
        sigma=1.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "time" in df.columns
    assert "status" in df.columns
    assert "X0" in df.columns
    assert "X1" in df.columns
    assert set(df["status"].unique()).issubset({0, 1})