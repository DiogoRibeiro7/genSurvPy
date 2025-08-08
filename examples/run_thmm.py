import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from gen_surv import generate

df = generate(
    model="thmm",
    n=100,
    qmat=[[0, 0.2, 0], [0.1, 0, 0.1], [0, 0.3, 0]],
    emission_pars={"mu": [0.0, 1.0, 2.0], "sigma": [0.5, 0.5, 0.5]},
    p0=[1.0, 0.0, 0.0],
    model_cens="exponential",
    cens_par=3.0,
    seed=42
)

print(df.head())
