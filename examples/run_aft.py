import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gen_surv.interface import generate

# Generate synthetic survival data using Log-Normal AFT model
df = generate(
    model="aft_ln",
    n=100,
    beta=[0.5, -0.3],
    sigma=1.0,
    model_cens="exponential",
    cens_par=3.0,
    seed=123
)

print(df.head())
