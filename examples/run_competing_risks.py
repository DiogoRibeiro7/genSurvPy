"""
Example demonstrating the Competing Risks models and visualization.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gen_surv import generate
from gen_surv.competing_risks import (
    cause_specific_cumulative_incidence,
    gen_competing_risks,
    gen_competing_risks_weibull,
)
from gen_surv.summary import compare_survival_datasets, summarize_survival_dataset


def plot_cause_specific_cumulative_incidence(df, time_points=None, figsize=(10, 6)):
    """Plot the cause-specific cumulative incidence functions."""
    if time_points is None:
        max_time = df["time"].max()
        time_points = np.linspace(0, max_time, 100)

    # Get unique causes (excluding censoring)
    causes = sorted([c for c in df["status"].unique() if c > 0])

    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)

    for cause in causes:
        cif = cause_specific_cumulative_incidence(df, time_points, cause=cause)
        ax.plot(cif["time"], cif["incidence"], label=f"Cause {cause}")

    # Add overlay showing number of subjects at each time
    time_bins = np.linspace(0, df["time"].max(), 10)
    event_counts = np.histogram(df.loc[df["status"] > 0, "time"], bins=time_bins)[0]

    # Add a secondary y-axis for event counts
    ax2 = ax.twinx()
    ax2.bar(
        time_bins[:-1],
        event_counts,
        width=time_bins[1] - time_bins[0],
        alpha=0.2,
        color="gray",
        align="edge",
    )
    ax2.set_ylabel("Number of events")
    ax2.grid(False)

    # Format the main plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Incidence")
    ax.set_title("Cause-Specific Cumulative Incidence Functions")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig, ax


# 1. Generate data with 2 competing risks
print("Generating data with exponential hazards...")
data_exponential = gen_competing_risks(
    n=500,
    n_risks=2,
    baseline_hazards=[0.5, 0.3],
    betas=[[0.8, -0.5], [0.2, 0.7]],
    model_cens="uniform",
    cens_par=2.0,
    seed=42,
)

# 2. Generate data with Weibull hazards (different shapes)
print("Generating data with Weibull hazards...")
data_weibull = gen_competing_risks_weibull(
    n=500,
    n_risks=2,
    shape_params=[0.8, 1.5],  # Decreasing vs increasing hazard
    scale_params=[2.0, 3.0],
    betas=[[0.8, -0.5], [0.2, 0.7]],
    model_cens="uniform",
    cens_par=2.0,
    seed=42,
)

# 3. Print summary statistics for both datasets
print("\nSummary of Exponential Hazards dataset:")
summarize_survival_dataset(data_exponential)

print("\nSummary of Weibull Hazards dataset:")
summarize_survival_dataset(data_weibull)

# 4. Compare event distributions
print("\nEvent distribution (Exponential Hazards):")
print(data_exponential["status"].value_counts())

print("\nEvent distribution (Weibull Hazards):")
print(data_weibull["status"].value_counts())

# 5. Plot cause-specific cumulative incidence functions
print("\nPlotting cumulative incidence functions...")
time_points = np.linspace(0, 5, 100)

fig1, ax1 = plot_cause_specific_cumulative_incidence(
    data_exponential, time_points=time_points, figsize=(10, 6)
)
plt.title("Cumulative Incidence Functions (Exponential Hazards)")
plt.savefig("cr_exponential_cif.png", dpi=300, bbox_inches="tight")

fig2, ax2 = plot_cause_specific_cumulative_incidence(
    data_weibull, time_points=time_points, figsize=(10, 6)
)
plt.title("Cumulative Incidence Functions (Weibull Hazards)")
plt.savefig("cr_weibull_cif.png", dpi=300, bbox_inches="tight")

# 6. Demonstrate using the unified generate() interface
print("\nUsing the unified generate() interface:")
data_unified = generate(
    model="competing_risks",
    n=100,
    n_risks=2,
    baseline_hazards=[0.5, 0.3],
    betas=[[0.8, -0.5], [0.2, 0.7]],
    model_cens="uniform",
    cens_par=2.0,
    seed=42,
)
print(data_unified.head())

# 7. Compare datasets
print("\nComparing datasets:")
comparison = compare_survival_datasets(
    {"Exponential": data_exponential, "Weibull": data_weibull}
)
print(comparison)

# Show plots if running interactively
if __name__ == "__main__":
    plt.show()
