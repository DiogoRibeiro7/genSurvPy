"""Analyse the pilot to decide the production design.

    python scripts/make_config.py --pilot
    python scripts/run_simulation.py --out results/raw/pilot.parquet
    python scripts/run_pilot.py --raw results/raw/pilot.parquet

The pilot exists to answer design questions, not scientific ones. Section 8 of
the protocol lists them, and this reports on each:

* which factors actually move the outcome, so redundant levels can be dropped;
* which cells are pathological or infeasible;
* where estimators fail, and how often;
* whether the censoring calibration held in the realised samples;
* how many replications the required Monte Carlo precision implies;
* what the production run would cost.

**Pilot results must never be pooled with production results.** They come from
a different design, usually a different number of replications, and they have
been looked at -- which is exactly what a preregistered protocol is meant to
prevent for the results that get reported. Everything written here goes to
``results/processed/pilot_*`` and nothing else reads it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from survival_misspec.aggregation import (  # noqa: E402
    aggregate,
    failure_rates,
    headline_metric_gap,
    read_raw,
    replications_for_precision,
)
from survival_misspec.hypotheses import analyse_hypotheses  # noqa: E402

#: Historical MISE-scale pilot diagnostic. The freeze decision is supported by
#: the explicit RMISE/H1-H4/headline precision report written below, without
#: automatically changing the planned production replication count.
TARGET_MCSE_MISE = 0.001
PRODUCTION_REPLICATIONS = 500


def _project_summary_mcse(
    summary: pd.DataFrame, target_replications: int
) -> pd.DataFrame:
    """Scale cell MCSE columns to a prospective production replication count."""
    projected = summary.copy()
    if (
        "c_index_harrell_integrated_hazard_mean" not in projected.columns
        and "c_index_harrell_mean" in projected.columns
    ):
        projected["c_index_harrell_integrated_hazard_mean"] = projected[
            "c_index_harrell_mean"
        ]
        if "c_index_harrell_mcse" in projected.columns:
            projected["c_index_harrell_integrated_hazard_mcse"] = projected[
                "c_index_harrell_mcse"
            ]

    for column in [name for name in projected.columns if name.endswith("_mcse")]:
        count_column = column.removesuffix("_mcse") + "_n"
        if count_column in projected.columns:
            counts = pd.to_numeric(projected[count_column], errors="coerce")
        else:
            counts = pd.Series(target_replications, index=projected.index)
        projected[column] = pd.to_numeric(projected[column], errors="coerce") * np.sqrt(
            counts / target_replications
        )
    return projected


def section(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default=str(HERE.parent / "results" / "raw" / "pilot.parquet")
    )
    parser.add_argument("--out", default=str(HERE.parent / "results" / "processed"))
    arguments = parser.parse_args()

    raw = read_raw(arguments.raw)
    if raw.empty:
        print(f"no pilot results at {arguments.raw}")
        return 1

    if raw.get("is_production", pd.Series(False)).any():
        print("REFUSING: this file contains production rows, not pilot rows.")
        return 1

    out = Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)

    section("Pilot coverage")
    print(f"  rows            {len(raw):,}")
    print(f"  scenarios       {raw['scenario_id'].nunique()}")
    print(f"  estimators      {raw['estimator_id'].nunique()}")
    print(f"  replications    {raw['replication_id'].nunique()}")

    # ---------------------------------------------------------------- failures
    section("Fit and scoring failures")
    failures = failure_rates(raw)
    bad = failures[(failures["fit_failures"] > 0) | (failures["score_failures"] > 0)]
    if bad.empty:
        print("  none: every estimator fitted and scored in every cell")
    else:
        for row in bad.itertuples():
            print(
                f"  {row.scenario_id:38} {row.estimator_id:24} "
                f"fit {row.fit_failure_rate:.0%}  score {row.score_failure_rate:.0%}"
            )
            if row.fit_error_types:
                print(f"      {row.fit_error_types[:120]}")
    failures.to_parquet(out / "pilot_failures.parquet", index=False)

    # ------------------------------------------------------- censoring control
    section("Censoring: target against realised")
    control = (
        raw.groupby(["dgp", "target_censoring"])["realised_censoring"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    control["error"] = (control["mean"] - control["target_censoring"]).abs()
    for row in control.sort_values("error", ascending=False).itertuples():
        flag = "  <-- off target" if row.error > 0.03 else ""
        print(
            f"  {row.dgp:24} target {row.target_censoring:.0%}  "
            f"realised {row.mean:.3f} (sd {row.std:.3f}){flag}"
        )

    # ------------------------------------------------------------ what matters
    section("Which factors move the primary loss (RMISE)")
    scored = raw[raw["scored"].fillna(False)]
    for factor in ("dgp", "n", "target_censoring", "estimator_id"):
        grouped = (
            scored.groupby(factor)["root_mean_integrated_squared_error"]
            .mean()
            .sort_values()
        )
        spread = grouped.max() / grouped.min() if grouped.min() > 0 else float("inf")
        print(f"\n  by {factor}  (max/min = {spread:.1f}x)")
        for key, value in grouped.items():
            print(f"    {str(key):34} {value:.6f}")

    # -------------------------------------------------- the paper's phenomenon
    section("Discrimination against truth recovery")
    print("  Does high concordance coexist with poor absolute survival prediction?")
    summary = aggregate(raw)
    c_index_column = (
        "c_index_harrell_integrated_hazard_mean"
        if "c_index_harrell_integrated_hazard_mean" in summary.columns
        else "c_index_harrell_mean"
    )
    view = summary[
        [
            "dgp",
            "estimator_id",
            c_index_column,
            "root_mean_integrated_squared_error_mean",
            "mean_absolute_survival_error_mean",
            "calibration_error_mean",
        ]
    ].rename(columns={c_index_column: "c_index"})
    view = view.dropna(subset=["c_index"])
    print(
        f"\n  {'dgp':24} {'estimator':24} {'C':>7} {'RMISE':>10} {'MAE_S':>8} {'calib':>8}"
    )
    for row in view.sort_values(
        ["dgp", "root_mean_integrated_squared_error_mean"]
    ).itertuples():
        print(
            f"  {row.dgp:24} {row.estimator_id:24} {row.c_index:7.4f} "
            f"{row.root_mean_integrated_squared_error_mean:10.6f} "
            f"{row.mean_absolute_survival_error_mean:8.4f} "
            f"{row.calibration_error_mean:8.4f}"
        )

    correlation = (
        view[["c_index", "root_mean_integrated_squared_error_mean"]].corr().iloc[0, 1]
    )
    print(f"\n  correlation(C-index, RMISE) across cells = {correlation:+.3f}")
    print("  A weak or positive correlation is the paper's phenomenon: ranking")
    print("  well and predicting probabilities well are not the same thing.")

    # ------------------------------------------------------------- replications
    section("Replications required for the target Monte Carlo precision")
    print(f"  target MCSE on MISE = {TARGET_MCSE_MISE}")
    required = []
    for (scenario, estimator), block in scored.groupby(["scenario_id", "estimator_id"]):
        values = block["mise"].to_numpy()
        need = replications_for_precision(values, TARGET_MCSE_MISE)
        required.append(
            {
                "scenario_id": scenario,
                "estimator_id": estimator,
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan,
                "required_R": need,
            }
        )
    requirement = pd.DataFrame(required)
    requirement.to_parquet(out / "pilot_replications.parquet", index=False)

    print(f"  median required R   {requirement['required_R'].median():.0f}")
    print(f"  90th percentile     {requirement['required_R'].quantile(0.90):.0f}")
    print(f"  maximum             {requirement['required_R'].max():.0f}")
    worst = requirement.nlargest(5, "required_R")
    print("\n  most demanding cells:")
    for row in worst.itertuples():
        print(f"    {row.scenario_id:38} {row.estimator_id:24} R>={row.required_R}")

    section("Projected production precision at R=500")
    precision_rows = []
    for (scenario, estimator), block in scored.groupby(["scenario_id", "estimator_id"]):
        values = pd.to_numeric(
            block["root_mean_integrated_squared_error"], errors="coerce"
        ).dropna()
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        precision_rows.append(
            {
                "scenario_id": scenario,
                "estimator_id": estimator,
                "pilot_replications": int(len(values)),
                "rmise_sd": sd,
                "expected_rmise_mcse_at_R500": (
                    sd / math.sqrt(PRODUCTION_REPLICATIONS)
                    if np.isfinite(sd)
                    else float("nan")
                ),
            }
        )
    precision = pd.DataFrame.from_records(precision_rows)
    precision.to_parquet(out / "pilot_precision.parquet", index=False)

    projected_summary = _project_summary_mcse(summary, PRODUCTION_REPLICATIONS)
    hypotheses = analyse_hypotheses(
        projected_summary, uncertainty_draws=500, seed=20260830
    )
    headline = headline_metric_gap(
        projected_summary, uncertainty_draws=500, seed=20260830
    )
    report = {
        "target_replications": PRODUCTION_REPLICATIONS,
        "rmise_mcse": {
            "median": float(
                precision["expected_rmise_mcse_at_R500"].median(skipna=True)
            ),
            "p90": float(precision["expected_rmise_mcse_at_R500"].quantile(0.90)),
            "max": float(precision["expected_rmise_mcse_at_R500"].max(skipna=True)),
        },
        "hypotheses": hypotheses[
            [
                "hypothesis",
                "estimate",
                "estimate_se",
                "estimate_ci_low",
                "estimate_ci_high",
                "n",
                "supports_hypothesis",
            ]
        ].to_dict("records"),
        "headline": {
            "max_loss_quantile_se": (
                float(headline["loss_quantile_se"].max(skipna=True))
                if not headline.empty
                else float("nan")
            ),
            "rows": int(len(headline)),
        },
    }
    (out / "pilot_precision_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        "  expected RMISE MCSE "
        f"median {report['rmise_mcse']['median']:.6f}, "
        f"p90 {report['rmise_mcse']['p90']:.6f}, "
        f"max {report['rmise_mcse']['max']:.6f}"
    )
    print("  derived estimands written to " f"{out / 'pilot_precision_report.json'}")

    # ------------------------------------------------------------------ cost
    section("Cost")
    per_cell = scored["fit_runtime_seconds"].mean()
    print(f"  mean fit time per cell   {per_cell:.3f} s")
    by_estimator = scored.groupby("estimator_id")["fit_runtime_seconds"].mean()
    for name, value in by_estimator.sort_values(ascending=False).items():
        print(f"    {name:26} {value:.3f} s")

    print("\n  Production cost scales as scenarios x estimators x R. The full")
    print("  Cartesian design in the protocol is 384 scenarios; the sections")
    print("  above are the evidence for reducing it.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
