import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate

df = generate(
    model="cmm",
    n=100,
    model_cens="exponential",
    cens_par=2.0,
    qmat=[[0, 0.1], [0.05, 0]],
    p0=[1.0, 0.0],
    seed=42,
)

print(df.head())
