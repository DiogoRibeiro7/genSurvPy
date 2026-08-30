"""Check truth-loss sensitivity to the evaluation grid resolution.

    python scripts/check_grid_convergence.py --config config --out results/grid_convergence.parquet

The production config uses 51 time points. This diagnostic reruns matched cells
at 51, 201 and 801 points and compares each with the 801-point value. The
default acceptance threshold is 0.002 RMISE on the survival-probability scale.
It also checks that the integrated-hazard Harrell C-index is numerically stable
against the finest grid.
The diagnostic only needs truth-loss metrics, so IPCW support-envelope
preparation is disabled by default for speed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from gate_artifacts import add_metadata, file_sha256, study_metadata  # noqa: E402
from survival_misspec.config import StudyConfig, load_study  # noqa: E402
from survival_misspec.estimators import fit_estimator  # noqa: E402
from survival_misspec.experiments import (  # noqa: E402
    EVALUATION_N,
    PreparedScenario,
    prepare_scenario,
)
from survival_misspec.metrics import (  # noqa: E402
    discrimination,
    expected_mortality,
    truth_recovery,
)
from survival_misspec.simulation import draw_replicate  # noqa: E402
from survival_misspec.truth import true_survival  # noqa: E402

DEFAULT_LOSS_COLUMNS = (
    "root_mean_integrated_squared_error_mean",
    "mise_mean",
)


def maximum_rmise_difference(frame: pd.DataFrame, reference_grid: int) -> float:
    """Maximum absolute RMISE difference among non-reference grids."""
    comparisons = frame[frame["n_time_points"] != reference_grid]
    if comparisons.empty:
        return float("nan")
    return float(
        pd.to_numeric(comparisons["rmise_absolute_difference"], errors="coerce").max()
    )


def maximum_c_index_difference(frame: pd.DataFrame, reference_grid: int) -> float:
    """Maximum absolute integrated-hazard C-index difference."""
    comparisons = frame[frame["n_time_points"] != reference_grid]
    if comparisons.empty:
        return float("nan")
    return float(
        pd.to_numeric(
            comparisons["c_index_harrell_integrated_hazard_absolute_difference"],
            errors="coerce",
        ).max()
    )


def grid_convergence_passes(
    frame: pd.DataFrame,
    *,
    reference_grid: int,
    rmise_epsilon: float,
    c_index_epsilon: float = 0.002,
) -> bool:
    """Return whether the pre-freeze grid-convergence criterion passes."""
    maximum_rmise = maximum_rmise_difference(frame, reference_grid)
    maximum_c_index = maximum_c_index_difference(frame, reference_grid)
    return bool(
        pd.notna(maximum_rmise)
        and maximum_rmise <= rmise_epsilon
        and pd.notna(maximum_c_index)
        and maximum_c_index <= c_index_epsilon
    )


def select_audit_cells(
    study: StudyConfig,
    *,
    summary: pd.DataFrame | None = None,
    top_cells: int | None = None,
    loss_column: str | None = None,
) -> set[tuple[str, str]] | None:
    """Scenario-estimator cells to audit, or ``None`` for the whole grid."""
    if top_cells is None:
        return None
    if top_cells <= 0:
        raise ValueError("top_cells must be positive")
    if summary is None:
        raise ValueError("top_cells requires a summary table")

    required = {"scenario_id", "estimator_id"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"summary missing columns: {missing}")

    candidates = [loss_column] if loss_column else list(DEFAULT_LOSS_COLUMNS)
    metric = next((column for column in candidates if column in summary.columns), None)
    if metric is None:
        raise ValueError(
            "summary does not contain any grid-audit loss column: "
            + ", ".join(candidates)
        )

    frame = summary.copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    available_scenarios = {scenario.scenario_id for scenario in study.scenarios}
    available_estimators = {estimator.estimator_id for estimator in study.estimators}
    frame = frame[
        frame["scenario_id"].isin(available_scenarios)
        & frame["estimator_id"].isin(available_estimators)
    ].dropna(subset=[metric])
    selected = frame.sort_values(metric, ascending=False).head(top_cells)
    return {
        (str(row.scenario_id), str(row.estimator_id)) for row in selected.itertuples()
    }


def score_replicate_across_grids(
    prepared_by_grid: dict[int, PreparedScenario],
    estimator,
    replication_id: int,
    master_seed: int,
) -> dict[int, dict[str, float]]:
    """Fit once, then evaluate truth loss on each candidate grid."""
    if not prepared_by_grid:
        return {}
    prepared = next(iter(prepared_by_grid.values()))
    scenario = prepared.config
    train = draw_replicate(
        scenario.dgp,
        prepared.params,
        scenario.n,
        scenario.scenario_id,
        replication_id,
        master_seed,
        stream="train",
    )
    evaluation = draw_replicate(
        scenario.dgp,
        prepared.params,
        EVALUATION_N,
        scenario.scenario_id,
        replication_id,
        master_seed,
        stream="eval",
    )
    fitted = fit_estimator(
        estimator.estimator_id,
        estimator.adapter,
        estimator.params,
        train.covariates,
        train.observed_time,
        train.event,
    )
    if not fitted.fitted:
        return {}

    by_grid: dict[int, dict[str, float]] = {}
    for points, grid_prepared in prepared_by_grid.items():
        grid = np.asarray(grid_prepared.time_grid, dtype=float)
        predicted = fitted.model.predict_survival(evaluation.covariates, grid)
        truth = true_survival(scenario.dgp, grid, evaluation.truth, prepared.params)
        row = truth_recovery(
            predicted,
            truth,
            grid,
            grid_prepared.tau,
        )
        row["c_index_harrell_integrated_hazard"] = discrimination(
            expected_mortality(predicted, grid),
            train.observed_time,
            train.event,
            evaluation.observed_time,
            evaluation.event,
            grid_prepared.tau,
        )["c_index_harrell"]
        by_grid[points] = row
    return by_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument(
        "--out", default=str(HERE.parent / "results" / "grid_convergence.parquet")
    )
    parser.add_argument("--grid-points", type=int, nargs="+", default=[51, 201, 801])
    parser.add_argument("--calibration-n", type=int, default=20000)
    parser.add_argument(
        "--ipcw-support-replications",
        type=int,
        default=0,
        help="support-envelope draws during preparation; 0 is enough for RMISE",
    )
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--estimators", nargs="*", default=None)
    parser.add_argument(
        "--summary",
        default=None,
        help="processed summary parquet used to select worst cells for the audit",
    )
    parser.add_argument(
        "--top-cells",
        type=int,
        default=None,
        help="audit only the top scenario-estimator cells by summary loss",
    )
    parser.add_argument(
        "--loss-column",
        default=None,
        help="summary column used with --top-cells; defaults to RMISE then MISE",
    )
    parser.add_argument(
        "--rmise-epsilon",
        type=float,
        default=0.002,
        help="maximum acceptable absolute RMISE difference versus the finest grid",
    )
    parser.add_argument(
        "--c-index-epsilon",
        type=float,
        default=0.002,
        help=(
            "maximum acceptable absolute integrated-hazard Harrell C-index "
            "difference versus the finest grid"
        ),
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
    summary_path = Path(arguments.summary) if arguments.summary is not None else None
    summary = pd.read_parquet(summary_path) if summary_path is not None else None
    selected_cells = select_audit_cells(
        study,
        summary=summary,
        top_cells=arguments.top_cells,
        loss_column=arguments.loss_column,
    )
    if selected_cells is not None:
        print(
            f"auditing top {len(selected_cells)} scenario-estimator cells", flush=True
        )

    metadata = study_metadata(study)
    metadata.update(
        {
            "audited_replications": arguments.replications,
            "rmise_epsilon": arguments.rmise_epsilon,
            "c_index_epsilon": arguments.c_index_epsilon,
            "selected_summary_sha256": (
                file_sha256(summary_path) if summary_path is not None else ""
            ),
            "top_cells": arguments.top_cells or 0,
            "loss_column": arguments.loss_column or "",
        }
    )

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        if selected_cells is not None and not any(
            scenario.scenario_id == scenario_id for scenario_id, _ in selected_cells
        ):
            continue
        prepared_by_grid = {}
        for points in grid_points:
            metrics = replace(study.metrics, n_time_points=points)
            prepared = prepare_scenario(
                scenario,
                metrics,
                calibration_n=arguments.calibration_n,
                ipcw_support_replications=arguments.ipcw_support_replications,
            )
            if prepared.feasible:
                prepared_by_grid[points] = prepared
        if reference_grid not in prepared_by_grid:
            continue

        for estimator in estimators:
            if (
                selected_cells is not None
                and (
                    scenario.scenario_id,
                    estimator.estimator_id,
                )
                not in selected_cells
            ):
                continue
            print(
                f"cell {scenario.scenario_id} / {estimator.estimator_id}",
                flush=True,
            )
            for replication_id in range(arguments.replications):
                by_grid = score_replicate_across_grids(
                    prepared_by_grid,
                    estimator,
                    replication_id,
                    study.master_seed,
                )
                if reference_grid not in by_grid:
                    continue
                reference = by_grid[reference_grid]
                for points, row in by_grid.items():
                    rows.append(
                        add_metadata(
                            {
                                "scenario_id": scenario.scenario_id,
                                "scenario_hash": scenario.hash,
                                "estimator_id": estimator.estimator_id,
                                "estimator_hash": estimator.hash,
                                "replication_id": replication_id,
                                "n_time_points": points,
                                "reference_n_time_points": reference_grid,
                                "mise": row["mise"],
                                "rmise": row["root_mean_integrated_squared_error"],
                                "c_index_harrell_integrated_hazard": row[
                                    "c_index_harrell_integrated_hazard"
                                ],
                                "mise_reference": reference["mise"],
                                "rmise_reference": reference[
                                    "root_mean_integrated_squared_error"
                                ],
                                "c_index_harrell_integrated_hazard_reference": reference[
                                    "c_index_harrell_integrated_hazard"
                                ],
                                "mise_absolute_difference": abs(
                                    row["mise"] - reference["mise"]
                                ),
                                "rmise_absolute_difference": abs(
                                    row["root_mean_integrated_squared_error"]
                                    - reference["root_mean_integrated_squared_error"]
                                ),
                                "c_index_harrell_integrated_hazard_absolute_difference": abs(
                                    row["c_index_harrell_integrated_hazard"]
                                    - reference["c_index_harrell_integrated_hazard"]
                                ),
                            },
                            metadata,
                        )
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
            [
                "mise_absolute_difference",
                "rmise_absolute_difference",
                "c_index_harrell_integrated_hazard_absolute_difference",
            ]
        ]
        .max()
    )
    print(summary.to_string())
    maximum_rmise = maximum_rmise_difference(frame, reference_grid)
    maximum_c_index = maximum_c_index_difference(frame, reference_grid)
    if grid_convergence_passes(
        frame,
        reference_grid=reference_grid,
        rmise_epsilon=arguments.rmise_epsilon,
        c_index_epsilon=arguments.c_index_epsilon,
    ):
        print(
            f"criterion pass: max |RMISE - RMISE_{reference_grid}| "
            f"{maximum_rmise:.6f} <= {arguments.rmise_epsilon:.6f}; "
            f"max |C - C_{reference_grid}| "
            f"{maximum_c_index:.6f} <= {arguments.c_index_epsilon:.6f}"
        )
        return 0
    print(
        f"criterion fail: max |RMISE - RMISE_{reference_grid}| "
        f"{maximum_rmise:.6f} > {arguments.rmise_epsilon:.6f} or "
        f"max |C - C_{reference_grid}| "
        f"{maximum_c_index:.6f} > {arguments.c_index_epsilon:.6f}"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
