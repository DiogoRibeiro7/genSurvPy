"""Audit whether production replications support the frozen IPCW grids.

    python scripts/audit_ipcw_availability.py --config config --out results/ipcw_availability.parquet

IBS and mean AUC are undefined when the prespecified IPCW grid falls outside a
replication's observed follow-up support. This pre-freeze audit estimates that
availability rate without fitting any models, because the support condition
depends only on the simulated train/evaluation outcome times.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.config import load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario  # noqa: E402
from survival_misspec.simulation import draw_replicate  # noqa: E402


def _supported(grid: np.ndarray, train_time: np.ndarray, eval_time: np.ndarray) -> bool:
    if grid.size < 2:
        return False
    lower = max(float(np.min(train_time)), float(np.min(eval_time)))
    upper = min(float(np.max(train_time)), float(np.max(eval_time)))
    return bool(np.all((grid > lower) & (grid < upper)))


def availability_failures(
    frame: pd.DataFrame, *, minimum_availability: float
) -> pd.DataFrame:
    """Feasible scenarios below the pre-freeze IPCW availability threshold."""
    feasible = frame[frame["feasible"]].copy()
    if feasible.empty:
        return feasible
    availability = pd.to_numeric(feasible["availability"], errors="coerce")
    return feasible[availability.isna() | (availability < minimum_availability)]


def availability_passes(frame: pd.DataFrame, *, minimum_availability: float) -> bool:
    """Return whether all feasible scenarios meet the IPCW availability gate."""
    return availability_failures(frame, minimum_availability=minimum_availability).empty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument(
        "--out", default=str(HERE.parent / "results" / "ipcw_availability.parquet")
    )
    parser.add_argument("--calibration-n", type=int, default=20000)
    parser.add_argument("--replications", type=int, default=50)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument(
        "--minimum-availability",
        type=float,
        default=0.95,
        help="minimum acceptable scenario-level availability rate",
    )
    arguments = parser.parse_args()

    study = load_study(arguments.config)
    rows: list[dict[str, object]] = []
    scenarios = list(study.scenarios)
    if arguments.max_scenarios is not None:
        scenarios = scenarios[: arguments.max_scenarios]

    for scenario in scenarios:
        prepared = prepare_scenario(
            scenario, study.metrics, calibration_n=arguments.calibration_n
        )
        if not prepared.feasible:
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "dgp": scenario.dgp,
                    "n": scenario.n,
                    "target_censoring": scenario.target_censoring,
                    "effect_size": scenario.effect_size,
                    "feasible": False,
                    "availability": float("nan"),
                    "supported": 0,
                    "attempted": 0,
                    "passes": False,
                    "reason": prepared.reason,
                }
            )
            continue

        grid = np.asarray(prepared.ipcw_time_grid, dtype=float)
        supported = 0
        for replication_id in range(arguments.replications):
            train = draw_replicate(
                scenario.dgp,
                prepared.params,
                scenario.n,
                scenario.scenario_id,
                replication_id,
                study.master_seed,
                stream="train",
            )
            evaluation = draw_replicate(
                scenario.dgp,
                prepared.params,
                scenario.n,
                scenario.scenario_id,
                replication_id,
                study.master_seed,
                stream="eval",
            )
            supported += int(
                _supported(grid, train.observed_time, evaluation.observed_time)
            )

        availability = supported / arguments.replications
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "dgp": scenario.dgp,
                "n": scenario.n,
                "target_censoring": scenario.target_censoring,
                "effect_size": scenario.effect_size,
                "feasible": True,
                "availability": availability,
                "supported": supported,
                "attempted": arguments.replications,
                "passes": availability >= arguments.minimum_availability,
                "reason": "",
            }
        )

    frame = pd.DataFrame.from_records(rows)
    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    feasible = frame[frame["feasible"]]
    failures = availability_failures(
        frame, minimum_availability=arguments.minimum_availability
    )
    print(f"written        {len(frame)} scenarios -> {out}")
    if not feasible.empty:
        print(f"minimum        {feasible['availability'].min():.3f}")
    if failures.empty:
        print(
            f"criterion      pass: all feasible scenarios >= {arguments.minimum_availability:.2f}"
        )
        return 0

    print(f"criterion      fail: {len(failures)} feasible scenarios below threshold")
    print(
        failures.sort_values("availability")[
            ["scenario_id", "availability", "n", "target_censoring", "effect_size"]
        ]
        .head(20)
        .to_string(index=False)
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
