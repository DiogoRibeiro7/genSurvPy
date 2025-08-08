import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_surv import generate

df = generate(
    model="cphm",
    n=100,
    model_cens="uniform",
    cens_par=1.0,
    beta=0.5,
    covar=2.0,
    seed=42
)

print(df.head())
