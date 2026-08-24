import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate

df = generate(
    model="tdcm",
    n=100,
    dist="weibull",
    corr=0.5,
    dist_par=[1, 2, 1, 2],
    model_cens="uniform",
    cens_par=1.0,
    # Two coefficients: the baseline covariate, and the effect of the
    # time-dependent one switching on. Passing three is deprecated.
    beta=[0.1, 0.2],
    lam=1.0,
    seed=42,
)

print(df.head())
