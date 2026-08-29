"""Executable definitions of the preregistered H1--H4 estimands."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

__all__ = [
    "CORRECTLY_SPECIFIED_REFERENCE",
    "PROPORTIONAL_HAZARDS_ESTIMATORS",
    "STRUCTURAL_VIOLATION_DGPS",
    "PH_OR_BASELINE_DGPS",
    "analyse_hypotheses",
]

CORRECTLY_SPECIFIED_REFERENCE = {
    "cphm": "cox_ph",
    "aft_weibull": "cox_ph",
    "piecewise_exponential": "cox_ph",
}
PROPORTIONAL_HAZARDS_ESTIMATORS = ("cox_ph", "gradient_boosted")
STRUCTURAL_VIOLATION_DGPS = ("aft_ln", "aft_log_logistic", "mixture_cure")
PH_OR_BASELINE_DGPS = ("cphm", "aft_weibull", "piecewise_exponential")

RMISE = "root_mean_integrated_squared_error_mean"
NMISE = "normalised_mise_mean"
C_INDEX = "c_index_harrell_mean"


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=list(columns))


def _record(
    hypothesis: str,
    estimand: str,
    estimate: float,
    *,
    supports: bool,
    n: int,
    criterion: str,
    note: str = "",
) -> dict[str, object]:
    return {
        "hypothesis": hypothesis,
        "estimand": estimand,
        "estimate": float(estimate) if np.isfinite(estimate) else float("nan"),
        "supports_hypothesis": bool(supports),
        "n": int(n),
        "criterion": criterion,
        "note": note,
    }


def _h1(summary: pd.DataFrame) -> dict[str, object]:
    frame = _finite(summary, [C_INDEX, RMISE])
    if "misspecification" in frame.columns:
        frame = frame[
            ~frame["misspecification"].astype(str).str.startswith("none", na=False)
        ]
    estimate = float(frame[C_INDEX].corr(frame[RMISE], method="spearman"))
    return _record(
        "H1",
        "spearman_c_index_rmise_misspecified_cells",
        estimate,
        supports=abs(estimate) < 0.30,
        n=len(frame),
        criterion="abs(estimate) < 0.30",
        note="computed on scenario-estimator cell means for misspecified scenarios",
    )


def _h2(summary: pd.DataFrame) -> dict[str, object]:
    frame = _finite(summary, [C_INDEX, NMISE])
    records = []
    for dgp, reference in CORRECTLY_SPECIFIED_REFERENCE.items():
        block = frame[frame["dgp"] == dgp]
        refs = block[block["estimator_id"] == reference][
            ["scenario_id", C_INDEX, NMISE]
        ].rename(
            columns={
                C_INDEX: "reference_c_index",
                NMISE: "reference_nmise",
            }
        )
        candidates = block[block["estimator_id"] != reference]
        joined = candidates.merge(refs, on="scenario_id", how="inner")
        joined = joined[joined["reference_nmise"] > 0]
        if not joined.empty:
            joined = joined.assign(
                nmise_ratio=joined[NMISE] / joined["reference_nmise"]
            )
            records.append(joined)

    if not records:
        return _record(
            "H2",
            "max_nmise_ratio_at_or_above_correct_reference_c_index",
            float("nan"),
            supports=False,
            n=0,
            criterion="estimate >= 10",
            note="no DGP had a mapped correctly specified reference estimator",
        )

    comparisons = pd.concat(records, ignore_index=True)
    admissible = comparisons[comparisons[C_INDEX] >= comparisons["reference_c_index"]]
    estimate = (
        float(admissible["nmise_ratio"].max()) if not admissible.empty else float("nan")
    )
    return _record(
        "H2",
        "max_nmise_ratio_at_or_above_correct_reference_c_index",
        estimate,
        supports=np.isfinite(estimate) and estimate >= 10.0,
        n=len(admissible),
        criterion="estimate >= 10",
        note=(
            "correct references are "
            + ", ".join(
                f"{dgp}:{estimator}"
                for dgp, estimator in CORRECTLY_SPECIFIED_REFERENCE.items()
            )
        ),
    )


def _h3(summary: pd.DataFrame) -> dict[str, object]:
    frame = _finite(summary, [RMISE])
    frame = frame[frame["estimator_id"].isin(PROPORTIONAL_HAZARDS_ESTIMATORS)]
    frame = frame[
        frame["dgp"].isin(STRUCTURAL_VIOLATION_DGPS)
        | frame["dgp"].isin(PH_OR_BASELINE_DGPS)
    ].copy()
    frame["group"] = np.where(
        frame["dgp"].isin(STRUCTURAL_VIOLATION_DGPS), "structural", "ph_or_baseline"
    )

    key = ["n", "target_censoring", "effect_size", "estimator_id"]
    rows = []
    for values, block in frame.groupby(key, dropna=False):
        groups = set(block["group"])
        if {"structural", "ph_or_baseline"} <= groups:
            means = block.groupby("group")[RMISE].mean()
            rows.append(
                {
                    **dict(
                        zip(key, values if isinstance(values, tuple) else (values,))
                    ),
                    "difference": float(means["structural"] - means["ph_or_baseline"]),
                }
            )
    paired = pd.DataFrame.from_records(rows)
    estimate = float(paired["difference"].mean()) if not paired.empty else float("nan")
    return _record(
        "H3",
        "common_support_structural_minus_ph_or_baseline_rmise",
        estimate,
        supports=np.isfinite(estimate) and estimate > 0.0,
        n=len(paired),
        criterion="estimate > 0",
        note=(
            "PH estimators: "
            + ", ".join(PROPORTIONAL_HAZARDS_ESTIMATORS)
            + "; structural DGPs: "
            + ", ".join(STRUCTURAL_VIOLATION_DGPS)
        ),
    )


def _h4(summary: pd.DataFrame) -> dict[str, object]:
    frame = _finite(summary, [RMISE, C_INDEX])
    low = frame[frame["target_censoring"] == 0.1]
    high = frame[frame["target_censoring"] == 0.7]
    key = ["dgp", "n", "effect_size", "estimator_id"]
    joined = high[key + [RMISE, C_INDEX]].merge(
        low[key + [RMISE, C_INDEX]],
        on=key,
        how="inner",
        suffixes=("_c70", "_c10"),
    )
    rmise_scale = float(frame[RMISE].std(ddof=1))
    c_index_scale = float(frame[C_INDEX].std(ddof=1))
    if joined.empty or rmise_scale <= 0 or c_index_scale <= 0:
        estimate = float("nan")
    else:
        rmise_degradation = (
            joined[f"{RMISE}_c70"] - joined[f"{RMISE}_c10"]
        ) / rmise_scale
        c_index_degradation = (
            joined[f"{C_INDEX}_c10"] - joined[f"{C_INDEX}_c70"]
        ) / c_index_scale
        estimate = float(rmise_degradation.mean() - c_index_degradation.mean())

    return _record(
        "H4",
        "common_support_standardised_rmise_degradation_minus_c_index_degradation",
        estimate,
        supports=np.isfinite(estimate) and estimate > 0.0,
        n=len(joined),
        criterion="estimate > 0",
        note="paired on DGP, n, effect size and estimator; missing 10% cure cells drop out",
    )


def analyse_hypotheses(summary: pd.DataFrame) -> pd.DataFrame:
    """Return one machine-readable row per preregistered hypothesis."""
    required = {"scenario_id", "dgp", "estimator_id", RMISE, C_INDEX}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"hypothesis analysis missing columns: {missing}")
    return pd.DataFrame.from_records(
        [_h1(summary), _h2(summary), _h3(summary), _h4(summary)]
    )
