from hypothesis import given, strategies as st
from gen_surv.aft import gen_aft_log_normal

@given(
    n=st.integers(min_value=1, max_value=20),
    sigma=st.floats(min_value=0.1, max_value=2.0, allow_nan=False, allow_infinity=False),
    cens_par=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=1000)
)
def test_gen_aft_log_normal_properties(n, sigma, cens_par, seed):
    df = gen_aft_log_normal(
        n=n,
        beta=[0.5, -0.2],
        sigma=sigma,
        model_cens="uniform",
        cens_par=cens_par,
        seed=seed
    )
    assert df.shape[0] == n
    assert set(df["status"].unique()).issubset({0, 1})
    assert (df["time"] >= 0).all()
    assert df.filter(regex="^X[0-9]+$").shape[1] == 2
