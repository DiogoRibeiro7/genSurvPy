"""Reproduce the entire study from a clean checkout, in one command.

    python scripts/reproduce.py --pilot
    python scripts/reproduce.py --production --lock protocol/experiment_lock.json --workers 8

Runs configuration, simulation, aggregation, tables and figures in order, and
stops at the first failure rather than continuing with a stale artefact.

This exists because "the experiment is reproducible" is a claim, and a claim
nobody executes is one nobody has checked. The final step of the protocol is to
reproduce the whole thing from the frozen commit; this is the command that does
it, so the claim is testable rather than aspirational.

**Production requires a lock.** Without one the run is refused rather than
quietly producing results that cannot be traced to a frozen experiment. With
one, `run_simulation.py` verifies the commit, the dependency versions and the
design hash before drawing a single sample.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def step(description: str, command: list[str]) -> None:
    print(f"\n{'=' * 74}\n{description}\n{'=' * 74}")
    print("  $ " + " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        print(f"\nFAILED after {elapsed:.1f}s: {description}")
        print("Stopping. Later steps would run on stale or missing inputs.")
        raise SystemExit(result.returncode)
    print(f"  done in {elapsed:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--production", action="store_true")
    parser.add_argument("--lock", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="reuse existing raw results and redo only the analysis",
    )
    arguments = parser.parse_args()

    python = sys.executable
    mode = "pilot" if arguments.pilot else "production"
    raw = f"results/raw/{mode}.parquet"

    if arguments.production and not arguments.lock:
        print(
            "REFUSING: a production run needs --lock. Without it the results "
            "cannot be tied to a frozen experiment, which is the whole point "
            "of freezing one."
        )
        return 1

    step(
        "1. Generate the configuration",
        [python, "scripts/make_config.py", f"--{mode}"],
    )

    if not arguments.skip_simulation:
        command = [
            python,
            "scripts/run_simulation.py",
            "--out",
            raw,
            "--workers",
            str(arguments.workers),
        ]
        if arguments.lock:
            command += ["--lock", arguments.lock]
        step("2. Run the Monte Carlo simulation", command)
    else:
        print("\n2. Simulation skipped (--skip-simulation)")

    step(
        "3. Aggregate, with Monte Carlo standard errors",
        [python, "scripts/aggregate_results.py", "--raw", raw],
    )
    step(
        "4. Analyse the preregistered hypotheses",
        [python, "scripts/analyze_hypotheses.py"],
    )
    step("5. Generate the tables", [python, "scripts/make_tables.py"])
    step(
        "6. Generate the figures",
        [python, "scripts/make_figures.py", "--raw", raw],
    )

    if arguments.pilot:
        step(
            "7. Report the pilot design analysis",
            [python, "scripts/run_pilot.py", "--raw", raw],
        )

    print(f"\n{'=' * 74}")
    print(f"{mode} pipeline complete.")
    if arguments.pilot:
        print("These are exploratory results. Figures carry a PILOT stamp and")
        print("must not be reported. Pilot artefacts are gitignored.")
    print(f"{'=' * 74}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
