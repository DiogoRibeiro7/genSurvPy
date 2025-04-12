import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gen_surv.cphm import gen_cphm

def test_gen_cphm_output_shape():
    df = gen_cphm(n=50, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)
    assert df.shape == (50, 3)
    assert list(df.columns) == ["time", "status", "covariate"]

def test_gen_cphm_status_range():
    df = gen_cphm(n=100, model_cens="exponential", cens_par=0.8, beta=0.3, covar=1.5)
    assert df["status"].isin([0, 1]).all()
