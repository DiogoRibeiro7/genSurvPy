"""
Example demonstrating Weibull AFT model and visualization capabilities.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate
from gen_surv.visualization import (
    describe_survival,
    plot_covariate_effect,
    plot_hazard_comparison,
    plot_survival_curve,
)

# 1. Generate data from different models for comparison
models = {
    "Weibull AFT (shape=0.5)": generate(
        model="aft_weibull",
        n=200,
        beta=[0.5, -0.3],
        shape=0.5,  # Decreasing hazard
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    ),
    "Weibull AFT (shape=1.0)": generate(
        model="aft_weibull",
        n=200,
        beta=[0.5, -0.3],
        shape=1.0,  # Constant hazard
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    ),
    "Weibull AFT (shape=2.0)": generate(
        model="aft_weibull",
        n=200,
        beta=[0.5, -0.3],
        shape=2.0,  # Increasing hazard
        scale=2.0,
        model_cens="uniform",
        cens_par=5.0,
        seed=42,
    ),
}

# Print sample data
print("Sample data from Weibull AFT model (shape=2.0):")
print(models["Weibull AFT (shape=2.0)"].head())
print("\n")

# 2. Compare survival curves from different models
fig1, ax1 = plot_survival_curve(
    data=pd.concat([df.assign(_model=name) for name, df in models.items()]),
    group_col="_model",
    title="Comparing Survival Curves with Different Weibull Shapes",
)
plt.savefig("survival_curve_comparison.png", dpi=300, bbox_inches="tight")

# 3. Compare hazard functions
fig2, ax2 = plot_hazard_comparison(
    models=models, title="Comparing Hazard Functions with Different Weibull Shapes"
)
plt.savefig("hazard_comparison.png", dpi=300, bbox_inches="tight")

# 4. Visualize covariate effect on survival
fig3, ax3 = plot_covariate_effect(
    data=models["Weibull AFT (shape=2.0)"],
    covariate_col="X0",
    n_groups=3,
    title="Effect of X0 Covariate on Survival",
)
plt.savefig("covariate_effect.png", dpi=300, bbox_inches="tight")

# 5. Summary statistics
for name, df in models.items():
    print(f"Summary for {name}:")
    summary = describe_survival(df)
    print(summary)
    print("\n")

print("Plots saved to current directory.")

# Show plots if running interactively
if __name__ == "__main__":
    plt.show()
