import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv.thmm import gen_thmm

def test_gen_thmm_shape():
    df = gen_thmm(
        n=50,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[0.5, 0.6, 0.7],
    )
    assert df.shape[1] == 4
    assert set(df["state"].unique()).issubset({1, 2, 3})
