"""Turn replicate rows into the tables the paper reports.

    python scripts/aggregate_results.py --raw results/raw/production.parquet

Writes to ``results/processed/``:

``summary.parquet``
    One row per (scenario, estimator): the mean of every metric with its Monte
    Carlo standard error and the number of replications it is based on.

``failures.parquet``
    Fit and scoring failure rates per cell, with the exception types. Kept
    separate so that a cell's metrics and its reliability are never read as one
    number.

``adequacy.parquet``
    Excess loss of each estimator over the reference, across a range of
    epsilon, for the adequacy region.

Nothing here decides anything. It aggregates and reports uncertainty; the
interpretation belongs in the paper, conditional on the DGPs studied.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.aggregation import (  # noqa: E402
    adequacy_region,
    aggregate,
    failure_rates,
    read_raw,
)

EPSILONS = (0.001, 0.005, 0.01, 0.025, 0.05)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default=str(HERE.parent / "results" / "processed"))
    parser.add_argument("--reference", default="cox_ph")
    arguments = parser.parse_args()

    raw = read_raw(arguments.raw)
    if raw.empty:
        print(f"no rows in {arguments.raw}")
        return 1

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"raw rows        {len(raw):,}")
    print(f"scenarios       {raw['scenario_id'].nunique()}")
    print(f"estimators      {raw['estimator_id'].nunique()}")
    if "is_production" in raw.columns:
        production = bool(raw["is_production"].any())
        print(f"production      {production}")
        if not production:
            print("                (exploratory run: not for publication)")

    summary = aggregate(raw)
    summary.to_parquet(out / "summary.parquet", index=False)
    print(f"summary         {len(summary)} cells -> {out / 'summary.parquet'}")

    failures = failure_rates(raw)
    failures.to_parquet(out / "failures.parquet", index=False)
    total_failures = int(
        failures["fit_failures"].sum() + failures["score_failures"].sum()
    )
    print(f"failures        {total_failures} across {len(failures)} cells")

    frames = []
    for epsilon in EPSILONS:
        try:
            frames.append(
                adequacy_region(summary, arguments.reference, epsilon=epsilon)
            )
        except ValueError as error:
            print(f"adequacy        skipped at epsilon={epsilon}: {error}")
            break
    if frames:
        adequacy = pd.concat(frames, ignore_index=True)
        adequacy.to_parquet(out / "adequacy.parquet", index=False)
        print(f"adequacy        {len(adequacy)} rows over {len(EPSILONS)} epsilons")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
