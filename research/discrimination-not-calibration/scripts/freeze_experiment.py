"""Freeze the production experiment design into an experiment lock.

    python scripts/freeze_experiment.py --config config --out protocol/experiment_lock.json

This command is the only supported way to create the production lock: it
prepares every scenario exactly once, records feasible and infeasible scenarios,
and writes the hash that production rows must carry when they are resumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from audit_ipcw_availability import availability_passes  # noqa: E402
from check_grid_convergence import (  # noqa: E402
    grid_convergence_passes,
    maximum_c_index_difference,
    maximum_rmise_difference,
    select_audit_cells,
)
from gate_artifacts import file_sha256, metadata_problems  # noqa: E402
from survival_misspec.config import StudyConfig, load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario  # noqa: E402
from survival_misspec.validation import write_lock  # noqa: E402


def _fail_if_any(problems: list[str], *, artifact: Path) -> None:
    if problems:
        raise SystemExit(
            f"gate artifact does not match the current production contract: {artifact}\n"
            + "\n".join(f"  - {problem}" for problem in problems)
        )


def _ipcw_gate_evidence(
    path: Path, minimum_availability: float, study: StudyConfig
) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"missing IPCW gate artifact: {path}")
    frame = pd.read_parquet(path)
    problems = metadata_problems(
        frame,
        study,
        expected={
            "audited_replications": study.n_replications,
            "minimum_availability_threshold": minimum_availability,
        },
    )
    scenario_ids = {scenario.scenario_id for scenario in study.scenarios}
    artifact_ids = set(frame.get("scenario_id", pd.Series(dtype=str)).astype(str))
    missing = sorted(scenario_ids - artifact_ids)
    extra = sorted(artifact_ids - scenario_ids)
    if missing:
        problems.append(f"IPCW artifact is missing scenarios: {missing[:10]}")
    if extra:
        problems.append(f"IPCW artifact contains unknown scenarios: {extra[:10]}")
    if "scenario_hash" in frame.columns:
        hashes = {
            str(row.scenario_id): row.scenario_hash
            for row in frame[["scenario_id", "scenario_hash"]].itertuples()
        }
        for scenario in study.scenarios:
            if hashes.get(scenario.scenario_id) != scenario.hash:
                problems.append(
                    f"scenario_hash mismatch for {scenario.scenario_id}: "
                    f"{hashes.get(scenario.scenario_id)} != {scenario.hash}"
                )
    else:
        problems.append("IPCW artifact missing scenario_hash")
    if "attempted" in frame.columns:
        feasible = frame[frame["feasible"]]
        attempted = pd.to_numeric(feasible["attempted"], errors="coerce")
        if not (attempted == study.n_replications).all():
            problems.append(
                "not every feasible scenario audited all planned replications"
            )
    else:
        problems.append("IPCW artifact missing attempted")
    _fail_if_any(problems, artifact=path)
    if not availability_passes(frame, minimum_availability=minimum_availability):
        raise SystemExit(
            f"IPCW availability gate failed in {path}; rerun "
            "scripts/audit_ipcw_availability.py before freezing"
        )
    feasible = frame[frame["feasible"]]
    return {
        "artifact": str(path),
        "sha256": file_sha256(path),
        "status": "PASS",
        "threshold": float(minimum_availability),
        "study_hash": study.hash,
        "git_commit": str(frame["git_commit"].iloc[0]),
        "scenario_design_hash": str(frame["scenario_design_hash"].iloc[0]),
        "estimator_design_hash": str(frame["estimator_design_hash"].iloc[0]),
        "metrics_hash": str(frame["metrics_hash"].iloc[0]),
        "audited_replications": study.n_replications,
        "minimum_availability": (
            float(feasible["availability"].min())
            if not feasible.empty
            else float("nan")
        ),
        "scenarios": int(len(frame)),
        "feasible_scenarios": int(len(feasible)),
    }


def _grid_gate_evidence(
    path: Path,
    rmise_epsilon: float,
    c_index_epsilon: float,
    study: StudyConfig,
    summary_path: Path,
    top_cells: int,
    minimum_replications: int,
) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"missing grid-convergence gate artifact: {path}")
    if not summary_path.exists():
        raise SystemExit(f"missing grid-convergence selection summary: {summary_path}")
    frame = pd.read_parquet(path)
    if frame.empty or "reference_n_time_points" not in frame.columns:
        raise SystemExit(f"grid-convergence gate artifact is incomplete: {path}")
    problems = metadata_problems(
        frame,
        study,
        expected={
            "rmise_epsilon": rmise_epsilon,
            "c_index_epsilon": c_index_epsilon,
            "selected_summary_sha256": file_sha256(summary_path),
            "top_cells": top_cells,
        },
    )
    summary = pd.read_parquet(summary_path)
    selected = select_audit_cells(study, summary=summary, top_cells=top_cells)
    artifact_cells = {
        (str(row.scenario_id), str(row.estimator_id))
        for row in frame[["scenario_id", "estimator_id"]].drop_duplicates().itertuples()
    }
    if artifact_cells != selected:
        problems.append(
            "grid artifact cells do not match the current selected worst cells"
        )
    if "replication_id" in frame.columns:
        replication_counts = frame.groupby(["scenario_id", "estimator_id"])[
            "replication_id"
        ].nunique()
        if (
            replication_counts.empty
            or (replication_counts < minimum_replications).any()
        ):
            problems.append(
                f"each grid cell must contain at least {minimum_replications} "
                "matched replications"
            )
    else:
        problems.append("grid artifact missing replication_id")
    if "audited_replications" in frame.columns:
        audited = pd.to_numeric(frame["audited_replications"], errors="coerce")
        if (audited < minimum_replications).any():
            problems.append(
                f"grid artifact audited fewer than {minimum_replications} replications"
            )
    else:
        problems.append("grid artifact missing audited_replications")
    _fail_if_any(problems, artifact=path)
    reference_grid = int(pd.to_numeric(frame["reference_n_time_points"]).max())
    maximum_rmise = maximum_rmise_difference(frame, reference_grid)
    maximum_c_index = maximum_c_index_difference(frame, reference_grid)
    if not grid_convergence_passes(
        frame,
        reference_grid=reference_grid,
        rmise_epsilon=rmise_epsilon,
        c_index_epsilon=c_index_epsilon,
    ):
        raise SystemExit(
            f"grid-convergence gate failed in {path}; rerun "
            "scripts/check_grid_convergence.py before freezing"
        )
    return {
        "artifact": str(path),
        "sha256": file_sha256(path),
        "status": "PASS",
        "rmise_threshold": float(rmise_epsilon),
        "c_index_threshold": float(c_index_epsilon),
        "study_hash": study.hash,
        "git_commit": str(frame["git_commit"].iloc[0]),
        "scenario_design_hash": str(frame["scenario_design_hash"].iloc[0]),
        "estimator_design_hash": str(frame["estimator_design_hash"].iloc[0]),
        "metrics_hash": str(frame["metrics_hash"].iloc[0]),
        "selected_summary_sha256": file_sha256(summary_path),
        "top_cells": top_cells,
        "audited_replications": minimum_replications,
        "maximum_rmise_difference": float(maximum_rmise),
        "maximum_c_index_difference": float(maximum_c_index),
        "reference_n_time_points": reference_grid,
        "rows": int(len(frame)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument(
        "--out", default=str(HERE.parent / "protocol" / "experiment_lock.json")
    )
    parser.add_argument("--protocol-version", default="0.1.0")
    parser.add_argument("--calibration-n", type=int, default=20000)
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--ipcw-gate",
        default=str(HERE.parent / "results" / "ipcw_availability.parquet"),
        help="passed IPCW availability gate artifact to embed in the lock",
    )
    parser.add_argument(
        "--grid-gate",
        default=str(HERE.parent / "results" / "grid_convergence.parquet"),
        help="passed grid-convergence gate artifact to embed in the lock",
    )
    parser.add_argument("--minimum-availability", type=float, default=0.95)
    parser.add_argument("--rmise-epsilon", type=float, default=0.002)
    parser.add_argument("--c-index-epsilon", type=float, default=0.002)
    parser.add_argument(
        "--grid-summary",
        default=str(HERE.parent / "results" / "processed" / "summary.parquet"),
        help="summary parquet whose worst cells selected the grid gate",
    )
    parser.add_argument("--grid-top-cells", type=int, default=10)
    parser.add_argument("--grid-minimum-replications", type=int, default=10)
    parser.add_argument(
        "--skip-gate-checks",
        action="store_true",
        help="write a dry-run lock without gate evidence",
    )
    parser.add_argument(
        "--allow-dirty-tree",
        action="store_true",
        help="permit a lock from uncommitted code for dry runs only",
    )
    arguments = parser.parse_args()

    study = load_study(arguments.config)
    gate_evidence = {}
    if not arguments.skip_gate_checks:
        gate_evidence = {
            "ipcw_availability": _ipcw_gate_evidence(
                Path(arguments.ipcw_gate), arguments.minimum_availability, study
            ),
            "grid_convergence": _grid_gate_evidence(
                Path(arguments.grid_gate),
                arguments.rmise_epsilon,
                arguments.c_index_epsilon,
                study,
                Path(arguments.grid_summary),
                arguments.grid_top_cells,
                arguments.grid_minimum_replications,
            ),
        }
    prepared = []
    print(f"study hash    {study.hash}")
    if gate_evidence:
        print("gates         PASS")
    print(f"scenarios     {len(study.scenarios)}")
    for scenario in study.scenarios:
        ready = prepare_scenario(
            scenario, study.metrics, calibration_n=arguments.calibration_n
        )
        prepared.append(ready.as_record())
        status = "OK" if ready.feasible else "SKIP"
        print(f"  {status} {ready.scenario_id}")

    lock = write_lock(
        arguments.out,
        study,
        prepared,
        arguments.protocol_version,
        notes=arguments.notes,
        allow_dirty_tree=arguments.allow_dirty_tree,
        gate_evidence=gate_evidence,
    )
    print(f"written       {arguments.out}")
    print(f"lock hash     {lock.lock_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
