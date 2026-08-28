"""Run the Monte Carlo study, or resume one that was interrupted.

    python scripts/run_simulation.py --out results/raw/pilot.parquet
    python scripts/run_simulation.py --out results/raw/production.parquet --lock protocol/experiment_lock.json

Resumption is by cell identity, never by count. Seeds are derived from
``(master_seed, scenario_id, replication_id, stream)``, so a resumed run
produces exactly the data an uninterrupted one would have -- which is what
makes it legitimate to stop and restart a production run at all.

With ``--lock`` the run refuses to start unless the current commit, the
dependency versions and the design hash all match the frozen experiment.
Without it, the run is exploratory and is marked as such in the output, so a
pilot cannot be mistaken for a production result later.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.aggregation import (  # noqa: E402
    completed_cells,
    write_raw,
)
from survival_misspec.config import load_study  # noqa: E402
from survival_misspec.experiments import prepare_scenario, run_cell  # noqa: E402
from survival_misspec.validation import (  # noqa: E402
    capture_provenance,
    verify_lock,
)

FLUSH_EVERY = 200


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--lock", default=None, help="verify against a frozen experiment"
    )
    parser.add_argument(
        "--max-cells", type=int, default=None, help="stop early (debugging)"
    )
    parser.add_argument("--calibration-n", type=int, default=20000)
    arguments = parser.parse_args()

    study = load_study(arguments.config)
    provenance = capture_provenance()

    print(f"paper_id      {study.paper_id}")
    print(f"study hash    {study.hash}")
    print(f"gen_surv      {provenance.gen_surv_version}")
    print(
        f"commit        {provenance.git_commit[:12]} "
        f"({'clean' if provenance.git_tree_clean else 'DIRTY'})"
    )
    print(
        f"design        {len(study.scenarios)} scenarios x "
        f"{len(study.estimators)} estimators x {study.n_replications} reps "
        f"= {study.n_cells:,} cells"
    )

    if arguments.lock:
        problems = verify_lock(arguments.lock, study, strict_commit=True)
        if problems:
            print("\nREFUSING TO RUN: this does not match the frozen experiment.")
            for problem in problems:
                print(f"  - {problem}")
            return 1
        print(f"lock          verified against {arguments.lock}")
    else:
        print("lock          NONE (exploratory run; not a production result)")

    out = Path(arguments.out)
    done = completed_cells(out)
    if done:
        print(f"resuming      {len(done):,} cells already present in {out}")

    print("\npreparing scenarios (calibrating censoring, fixing tau)...")
    prepared = []
    infeasible = []
    for scenario in study.scenarios:
        ready = prepare_scenario(
            scenario, study.metrics, calibration_n=arguments.calibration_n
        )
        if ready.feasible:
            prepared.append(ready)
        else:
            infeasible.append(ready)
            print(f"  SKIP {ready.scenario_id}: {ready.reason}")

    print(f"  {len(prepared)} feasible, {len(infeasible)} infeasible")
    if not prepared:
        print("nothing to run")
        return 1

    total = len(prepared) * len(study.estimators) * study.n_replications
    started = time.perf_counter()
    buffer: list[dict] = []
    completed = 0
    ran = 0

    for ready in prepared:
        for estimator in study.estimators:
            for replication_id in range(study.n_replications):
                key = (ready.scenario_id, estimator.estimator_id, replication_id)
                completed += 1
                if key in done:
                    continue

                row = run_cell(ready, estimator, replication_id, study.master_seed)
                row["study_hash"] = study.hash
                row["git_commit"] = provenance.git_commit
                row["gen_surv_version"] = provenance.gen_surv_version
                row["is_production"] = bool(arguments.lock)
                buffer.append(row)
                ran += 1

                if len(buffer) >= FLUSH_EVERY:
                    write_raw(buffer, out)
                    buffer = []
                    elapsed = time.perf_counter() - started
                    rate = ran / elapsed if elapsed else 0.0
                    remaining = (total - completed) / rate if rate else float("nan")
                    print(
                        f"  {completed:,}/{total:,}  ran {ran:,}  "
                        f"{rate:.1f} cells/s  eta {remaining / 60:.1f} min"
                    )

                if arguments.max_cells and ran >= arguments.max_cells:
                    write_raw(buffer, out)
                    print(f"stopped early at {ran} cells (--max-cells)")
                    return 0

    if buffer:
        write_raw(buffer, out)

    elapsed = time.perf_counter() - started
    print(f"\ndone: ran {ran:,} cells in {elapsed / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
