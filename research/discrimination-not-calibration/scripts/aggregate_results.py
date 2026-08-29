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

``paired_differences.parquet``
    Within-replication differences between each estimator and the reference.

``adequacy.parquet``
    Excess loss of each estimator over the reference, across a range of
    epsilon, for the adequacy region, with paired MCSEs.

``headline.parquet``
    Conditional upper-tail normalised truth loss among rows with comparable
    conventional metric values.

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
    adequacy_region_from_pairs,
    aggregate,
    failure_rates,
    headline_metric_gap,
    paired_differences,
    read_raw,
)

EPSILONS = (0.01, 0.025, 0.05, 0.10, 0.20)


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

    paired = paired_differences(raw, arguments.reference)
    paired.to_parquet(out / "paired_differences.parquet", index=False)
    print(f"paired diffs    {len(paired)} rows -> {out / 'paired_differences.parquet'}")

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
                adequacy_region_from_pairs(
                    paired,
                    loss="root_mean_integrated_squared_error",
                    epsilon=epsilon,
                )
            )
        except ValueError as error:
            print(f"adequacy        skipped at epsilon={epsilon}: {error}")
            break
    if frames:
        adequacy = pd.concat(frames, ignore_index=True)
        adequacy.to_parquet(out / "adequacy.parquet", index=False)
        print(f"adequacy        {len(adequacy)} rows over {len(EPSILONS)} epsilons")

    try:
        headline = headline_metric_gap(raw)
    except ValueError as error:
        print(f"headline        skipped: {error}")
    else:
        if headline.empty:
            print("headline        skipped: no scored rows")
        else:
            headline.to_parquet(out / "headline.parquet", index=False)
            print(f"headline        {len(headline)} bins -> {out / 'headline.parquet'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
