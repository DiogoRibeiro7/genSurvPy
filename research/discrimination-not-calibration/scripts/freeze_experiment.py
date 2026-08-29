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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.config import load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario  # noqa: E402
from survival_misspec.validation import write_lock  # noqa: E402


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
        "--allow-dirty-tree",
        action="store_true",
        help="permit a lock from uncommitted code for dry runs only",
    )
    arguments = parser.parse_args()

    study = load_study(arguments.config)
    prepared = []
    print(f"study hash    {study.hash}")
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
    )
    print(f"written       {arguments.out}")
    print(f"lock hash     {lock.lock_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
