"""Generate the YAML configuration from a declarative factorial design.

The scenario list is long enough that hand-writing it would guarantee typos and
make the design impossible to read as a design. This script emits
``config/simulation.yaml`` from the factor definitions below, so the YAML stays
explicit and diffable -- it is what the runs actually read -- while the
*intent* stays legible here.

Run it with ``--pilot`` for the reduced grid used to refine the design, or
``--production`` for the full one. Nothing here decides the production grid on
its own: §8 of the protocol requires the pilot to prune before production is
fixed, and :mod:`survival_misspec.simulation` reports infeasible cells rather
than approximating them.

    python scripts/make_config.py --pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / "config"

#: How each generator's parameters are built from a scalar effect size, so that
#: "effect_size" means the same knob across the design. The magnitudes are not
#: comparable *between* generators -- `cphm`'s covariate is Uniform(0, 2) and
#: the rest are standard normal -- and the paper compares within a DGP.
DGP_BUILDERS = {
    "cphm": lambda b: {
        "beta": b,
        "covariate_range": 2.0,
        "model_cens": "uniform",
    },
    "aft_weibull": lambda b: {
        "beta": [b, -b / 2],
        "shape": 1.5,
        "scale": 2.0,
        "model_cens": "uniform",
    },
    "aft_ln": lambda b: {
        "beta": [b, -b / 2],
        "sigma": 1.0,
        "model_cens": "uniform",
    },
    "aft_log_logistic": lambda b: {
        "beta": [b, -b / 2],
        "shape": 1.5,
        "scale": 2.0,
        "model_cens": "uniform",
    },
    "piecewise_exponential": lambda b: {
        "betas": [b, -b / 2],
        "breakpoints": [0.5, 1.5],
        "hazard_rates": [0.4, 0.9, 1.6],
        "model_cens": "uniform",
    },
    "mixture_cure": lambda b: {
        "betas_survival": [b, -b / 2],
        "betas_cure": [b / 2, b / 5],
        "cure_fraction": 0.3,
        "baseline_hazard": 0.7,
        "model_cens": "uniform",
    },
}

#: What each mechanism does to a proportional-hazards model. This is the axis
#: the paper is actually about, so it is recorded per scenario rather than
#: inferred from the generator name at analysis time.
MISSPECIFICATION = {
    "cphm": "none: proportional hazards, exponential baseline",
    "aft_weibull": "none for PH: monotone parametric baseline",
    "piecewise_exponential": "baseline shape only: PH holds, baseline is a step function",
    "aft_ln": "non-proportional hazards: log-normal, hazard rises then falls",
    "aft_log_logistic": "non-proportional hazards: unimodal hazard",
    "mixture_cure": "survival plateau: a cured fraction never fails",
}

PILOT = {
    "n": [250, 1000],
    "censoring": [0.3, 0.7],
    "effect": [0.5],
    "replications": 10,
}

#: The production grid, pruned from the full Cartesian design on pilot
#: evidence rather than by preference. The pilot measured how far each factor
#: moves the primary loss:
#:
#:     estimator 6.4x, dgp 6.1x, censoring 3.2x, n 1.7x
#:
#: so no factor is redundant and none is dropped. Two *levels* are, both
#: interior points on smooth trends: n=500 sits between 250 and 1000 on a 1.7x
#: range, and effect=0.25 between the null and 0.5.
#:
#: R=500 comes from inverting MCSE = s / sqrt(R) on pilot variability at a
#: target of 0.001 on MISE. That covers the 90th percentile of cells; the
#: hardest (aft_log_logistic at n=250 with 70% censoring) needs about 5000 and
#: stays underpowered. Its Monte Carlo error is reported rather than hidden,
#: and comparisons there are described as indistinguishable.
#:
#: mixture_cure loses its low-censoring cells automatically: a 30% cure
#: fraction puts a floor of about 31% on the censoring rate.
PRODUCTION = {
    "n": [250, 1000, 5000],
    "censoring": [0.1, 0.3, 0.5, 0.7],
    "effect": [0.0, 0.5, 1.0],
    "replications": 500,
}


def build_scenarios(grid: dict[str, Any]) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for dgp, builder in DGP_BUILDERS.items():
        for n in grid["n"]:
            for censoring in grid["censoring"]:
                for effect in grid["effect"]:
                    scenario_id = (
                        f"{dgp}__n{n}__c{int(round(censoring * 100)):02d}"
                        f"__b{effect:.2f}".replace(".", "p")
                    )
                    scenarios.append(
                        {
                            "scenario_id": scenario_id,
                            "dgp": dgp,
                            "n": n,
                            "target_censoring": censoring,
                            "effect_size": effect,
                            "misspecification": MISSPECIFICATION[dgp],
                            "params": builder(effect),
                        }
                    )
    return scenarios


ESTIMATORS = [
    {
        "estimator_id": "cox_ph",
        "adapter": "cox_ph",
        "params": {},
        "assumptions": (
            "Proportional hazards; semi-parametric baseline; linear in the "
            "covariates on the log-hazard scale."
        ),
    },
    {
        "estimator_id": "weibull_aft",
        "adapter": "weibull_aft",
        "params": {},
        "assumptions": (
            "Fully parametric Weibull; monotone hazard; linear in the "
            "covariates on the log-time scale."
        ),
    },
    {
        "estimator_id": "random_survival_forest",
        "adapter": "random_survival_forest",
        "params": {"n_estimators": 100, "min_samples_leaf": 15, "n_jobs": 1},
        "assumptions": (
            "None on the hazard's shape or proportionality; estimates a "
            "discrete hazard from the sample, so it needs data to do so."
        ),
    },
    {
        "estimator_id": "gradient_boosted",
        "adapter": "gradient_boosted",
        "params": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
        "assumptions": (
            "Proportional hazards through the partial-likelihood loss, but no "
            "assumption of linearity in the covariates."
        ),
    },
]

METRICS = {
    "tau_quantile": 0.80,
    "n_time_points": 51,
    "time_grid_quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "metrics": [
        "mise",
        "normalised_mise",
        "root_mean_integrated_squared_error",
        "miae",
        "mean_absolute_survival_error",
        "c_index_harrell",
        "c_index_uno",
        "c_index_at_tau",
        "c_index_antolini",
        "d_calibration_p",
        "brier_at_tau",
        "integrated_brier_score",
        "auc_mean",
        "calibration_error",
    ],
}


def write(grid: dict[str, Any], paper_id: str, master_seed: int) -> None:
    CONFIG.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios(grid)

    header = (
        "# Generated by scripts/make_config.py. Edit that script, not this file:\n"
        "# a hand edit here will be overwritten and will not be reflected in the\n"
        "# design it is supposed to express.\n"
    )

    simulation = {
        "paper_id": paper_id,
        "master_seed": master_seed,
        "n_replications": grid["replications"],
        "scenarios": scenarios,
    }
    (CONFIG / "simulation.yaml").write_text(
        header + yaml.safe_dump(simulation, sort_keys=False), encoding="utf-8"
    )
    (CONFIG / "estimators.yaml").write_text(
        header + yaml.safe_dump({"estimators": ESTIMATORS}, sort_keys=False),
        encoding="utf-8",
    )
    (CONFIG / "metrics.yaml").write_text(
        header + yaml.safe_dump(METRICS, sort_keys=False), encoding="utf-8"
    )

    cells = len(scenarios) * len(ESTIMATORS) * grid["replications"]
    print(f"  scenarios:    {len(scenarios)}")
    print(f"  estimators:   {len(ESTIMATORS)}")
    print(f"  replications: {grid['replications']}")
    print(f"  total cells:  {cells:,}")
    print(f"  written to:   {CONFIG}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--production", action="store_true")
    parser.add_argument("--master-seed", type=int, default=20260828)
    arguments = parser.parse_args()

    if arguments.pilot:
        write(PILOT, "discrimination-not-calibration-pilot", arguments.master_seed)
    else:
        write(PRODUCTION, "discrimination-not-calibration", arguments.master_seed)


if __name__ == "__main__":
    main()
