import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gen_surv.cmm import gen_cmm


def test_gen_cmm_shape():
    df = gen_cmm(
        n=50,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=2.0,
        rate=[0.1, 1.0, 0.2, 1.0, 0.3, 1.0],
    )
    assert df.shape[1] == 6
    assert "transition" in df.columns
