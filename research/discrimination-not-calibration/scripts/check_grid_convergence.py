"""Check truth-loss sensitivity to the evaluation grid resolution.

    python scripts/check_grid_convergence.py --config config --out results/grid_convergence.parquet

The production config uses 51 time points. This diagnostic reruns matched cells
at 51, 201 and 801 points and compares each with the 801-point value. The
default acceptance threshold is 0.005 RMISE, half a percentage point on the
survival-probability scale.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.config import load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario, run_cell  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument(
        "--out", default=str(HERE.parent / "results" / "grid_convergence.parquet")
    )
    parser.add_argument("--grid-points", type=int, nargs="+", default=[51, 201, 801])
    parser.add_argument("--calibration-n", type=int, default=20000)
    parser.add_argument("--replications", type=int, default=3)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument(
        "--rmise-epsilon",
        type=float,
        default=0.005,
        help="maximum acceptable absolute RMISE difference versus the finest grid",
    )
    arguments = parser.parse_args()

    study = load_study(arguments.config)
    grid_points = sorted(set(arguments.grid_points))
    if len(grid_points) < 2:
        raise SystemExit("need at least two grid sizes")
    reference_grid = grid_points[-1]

    scenarios = list(study.scenarios)
    if arguments.max_scenarios is not None:
        scenarios = scenarios[: arguments.max_scenarios]
    estimators = [
        estimator
        for estimator in study.estimators
        if arguments.estimators is None
        or estimator.estimator_id in set(arguments.estimators)
    ]

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        prepared_by_grid = {}
        for points in grid_points:
            metrics = replace(study.metrics, n_time_points=points)
            prepared = prepare_scenario(
                scenario, metrics, calibration_n=arguments.calibration_n
            )
            if prepared.feasible:
                prepared_by_grid[points] = prepared
        if reference_grid not in prepared_by_grid:
            continue

        for estimator in estimators:
            for replication_id in range(arguments.replications):
                by_grid = {}
                for points, prepared in prepared_by_grid.items():
                    row = run_cell(
                        prepared,
                        estimator,
                        replication_id,
                        study.master_seed,
                    )
                    if row.get("scored"):
                        by_grid[points] = row
                if reference_grid not in by_grid:
                    continue
                reference = by_grid[reference_grid]
                for points, row in by_grid.items():
                    rows.append(
                        {
                            "scenario_id": scenario.scenario_id,
                            "estimator_id": estimator.estimator_id,
                            "replication_id": replication_id,
                            "n_time_points": points,
                            "reference_n_time_points": reference_grid,
                            "mise": row["mise"],
                            "rmise": row["root_mean_integrated_squared_error"],
                            "mise_reference": reference["mise"],
                            "rmise_reference": reference[
                                "root_mean_integrated_squared_error"
                            ],
                            "mise_absolute_difference": abs(
                                row["mise"] - reference["mise"]
                            ),
                            "rmise_absolute_difference": abs(
                                row["root_mean_integrated_squared_error"]
                                - reference["root_mean_integrated_squared_error"]
                            ),
                        }
                    )

    if not rows:
        print("no scored grid-convergence rows")
        return 1

    out = Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(rows)
    frame.to_parquet(out, index=False)
    print(f"written {len(frame)} rows -> {out}")
    summary = (
        frame[frame["n_time_points"] != reference_grid]
        .groupby("n_time_points")[
            ["mise_absolute_difference", "rmise_absolute_difference"]
        ]
        .max()
    )
    print(summary.to_string())
    maximum = float(summary["rmise_absolute_difference"].max())
    if maximum <= arguments.rmise_epsilon:
        print(
            f"criterion pass: max |RMISE - RMISE_{reference_grid}| "
            f"{maximum:.6f} <= {arguments.rmise_epsilon:.6f}"
        )
        return 0
    print(
        f"criterion fail: max |RMISE - RMISE_{reference_grid}| "
        f"{maximum:.6f} > {arguments.rmise_epsilon:.6f}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
