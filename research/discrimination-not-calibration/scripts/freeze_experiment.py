"""Freeze the production experiment design into an experiment lock.

    python scripts/freeze_experiment.py --config config --out protocol/experiment_lock.json

This command is the only supported way to create the production lock: it
prepares every scenario exactly once, records feasible and infeasible scenarios,
and writes the hash that production rows must carry when they are resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from audit_ipcw_availability import availability_passes  # noqa: E402
from check_grid_convergence import (  # noqa: E402
    grid_convergence_passes,
    maximum_rmise_difference,
)
from survival_misspec.config import load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario  # noqa: E402
from survival_misspec.validation import write_lock  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ipcw_gate_evidence(path: Path, minimum_availability: float) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"missing IPCW gate artifact: {path}")
    frame = pd.read_parquet(path)
    if not availability_passes(frame, minimum_availability=minimum_availability):
        raise SystemExit(
            f"IPCW availability gate failed in {path}; rerun "
            "scripts/check_ipcw_availability.py before freezing"
        )
    feasible = frame[frame["feasible"]]
    return {
        "artifact": str(path),
        "sha256": _sha256(path),
        "status": "PASS",
        "threshold": float(minimum_availability),
        "minimum_availability": (
            float(feasible["availability"].min())
            if not feasible.empty
            else float("nan")
        ),
        "scenarios": int(len(frame)),
        "feasible_scenarios": int(len(feasible)),
    }


def _grid_gate_evidence(path: Path, rmise_epsilon: float) -> dict[str, object]:
    if not path.exists():
        raise SystemExit(f"missing grid-convergence gate artifact: {path}")
    frame = pd.read_parquet(path)
    if frame.empty or "reference_n_time_points" not in frame.columns:
        raise SystemExit(f"grid-convergence gate artifact is incomplete: {path}")
    reference_grid = int(pd.to_numeric(frame["reference_n_time_points"]).max())
    maximum = maximum_rmise_difference(frame, reference_grid)
    if not grid_convergence_passes(
        frame, reference_grid=reference_grid, rmise_epsilon=rmise_epsilon
    ):
        raise SystemExit(
            f"grid-convergence gate failed in {path}; rerun "
            "scripts/check_grid_convergence.py before freezing"
        )
    return {
        "artifact": str(path),
        "sha256": _sha256(path),
        "status": "PASS",
        "threshold": float(rmise_epsilon),
        "maximum_rmise_difference": float(maximum),
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
                Path(arguments.ipcw_gate), arguments.minimum_availability
            ),
            "grid_convergence": _grid_gate_evidence(
                Path(arguments.grid_gate), arguments.rmise_epsilon
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
