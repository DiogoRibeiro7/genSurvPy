"""Executable H1--H4 definitions."""

from __future__ import annotations

import pandas as pd
import pytest
from survival_misspec.hypotheses import analyse_hypotheses


def _summary() -> pd.DataFrame:
    rows = []
    for dgp, misspecification in (
        ("cphm", "none: proportional hazards, exponential baseline"),
        ("aft_weibull", "baseline shape only: PH holds"),
        ("piecewise_exponential", "baseline shape only: PH holds"),
        ("aft_ln", "non-proportional hazards"),
        ("aft_log_logistic", "non-proportional hazards"),
        ("mixture_cure", "survival plateau"),
    ):
        censoring_levels = [0.5, 0.7] if dgp == "mixture_cure" else [0.1, 0.5, 0.7]
        for censoring in censoring_levels:
            for estimator in ("cox_ph", "gradient_boosted", "random_survival_forest"):
                base = 0.02 if dgp in {"cphm", "piecewise_exponential"} else 0.20
                if estimator == "cox_ph" and dgp == "cphm":
                    base = 0.001
                if estimator == "random_survival_forest" and dgp == "cphm":
                    base = 0.02
                rmise = base + (0.04 if censoring == 0.7 else 0.0)
                c_index = 0.65 - (0.01 if censoring == 0.7 else 0.0)
                rows.append(
                    {
                        "scenario_id": f"{dgp}__c{int(censoring * 100)}",
                        "dgp": dgp,
                        "n": 250,
                        "target_censoring": censoring,
                        "effect_size": 0.5,
                        "estimator_id": estimator,
                        "misspecification": misspecification,
                        "c_index_harrell_mean": c_index,
                        "normalised_mise_mean": rmise**2,
                        "root_mean_integrated_squared_error_mean": rmise,
                    }
                )
    return pd.DataFrame.from_records(rows)


def test_hypotheses_are_machine_readable() -> None:
    hypotheses = analyse_hypotheses(_summary(), uncertainty_draws=20, seed=1)

    assert set(hypotheses["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    assert hypotheses["estimand"].notna().all()
    assert hypotheses["criterion"].notna().all()
    assert {
        "estimate_se",
        "estimate_ci_low",
        "estimate_ci_high",
        "bootstrap_mc_error",
    } <= set(hypotheses.columns)


def test_h2_uses_only_dgps_with_mapped_correct_references() -> None:
    h2 = (
        analyse_hypotheses(_summary(), uncertainty_draws=0)
        .set_index("hypothesis")
        .loc["H2"]
    )

    assert h2["estimate"] == pytest.approx(400.0)
    assert "aft_ln" not in h2["note"]
    assert bool(h2["supports_hypothesis"])


def test_h1_uses_spearman_not_pearson() -> None:
    summary = pd.DataFrame(
        {
            "scenario_id": [f"s{i}" for i in range(4)],
            "dgp": ["aft_ln"] * 4,
            "n": [250] * 4,
            "target_censoring": [0.5] * 4,
            "effect_size": [0.5] * 4,
            "estimator_id": ["cox_ph"] * 4,
            "misspecification": ["non-proportional hazards"] * 4,
            "c_index_harrell_mean": [0.60, 0.90, 0.70, 0.80],
            "normalised_mise_mean": [0.01, 0.0121, 0.0144, 0.25],
            "root_mean_integrated_squared_error_mean": [0.10, 0.11, 0.12, 0.50],
        }
    )

    h1 = (
        analyse_hypotheses(summary, uncertainty_draws=0)
        .set_index("hypothesis")
        .loc["H1"]
    )

    assert h1["estimate"] == pytest.approx(
        summary["c_index_harrell_mean"].corr(
            summary["root_mean_integrated_squared_error_mean"], method="spearman"
        )
    )
    assert h1["estimate"] != pytest.approx(
        summary["c_index_harrell_mean"].corr(
            summary["root_mean_integrated_squared_error_mean"], method="pearson"
        )
    )


def test_h3_and_h4_use_common_support() -> None:
    hypotheses = analyse_hypotheses(_summary(), uncertainty_draws=0).set_index(
        "hypothesis"
    )

    assert hypotheses.loc["H3", "n"] == 4
    assert hypotheses.loc["H4", "n"] == 15
    assert "common_support" in hypotheses.loc["H3", "estimand"]
    assert "common_support" in hypotheses.loc["H4", "estimand"]


def test_h4_excludes_mixture_cure_without_10_percent_support() -> None:
    h4 = (
        analyse_hypotheses(_summary(), uncertainty_draws=0)
        .set_index("hypothesis")
        .loc["H4"]
    )

    expected_pairs_without_cure = 5 * 3
    assert h4["n"] == expected_pairs_without_cure
    assert "missing 10% cure cells drop out" in h4["note"]


def test_h1_excludes_null_effect_cells_from_misspecification_primary() -> None:
    summary = _summary()
    null_rows = summary.copy()
    null_rows["scenario_id"] = null_rows["scenario_id"] + "__null"
    null_rows["effect_size"] = 0.0
    combined = pd.concat([summary, null_rows], ignore_index=True)

    h1 = (
        analyse_hypotheses(combined, uncertainty_draws=0)
        .set_index("hypothesis")
        .loc["H1"]
    )

    assert h1["n"] == len(summary[~summary["misspecification"].str.startswith("none")])
    assert "effect_size > 0" in h1["note"]


def test_h3_requires_complete_structural_and_baseline_support() -> None:
    summary = _summary()
    incomplete = summary[
        ~((summary["dgp"] == "mixture_cure") & (summary["target_censoring"] == 0.5))
    ]

    h3 = (
        analyse_hypotheses(incomplete, uncertainty_draws=0)
        .set_index("hypothesis")
        .loc["H3"]
    )

    assert h3["n"] == 2
    assert "complete structural" in h3["note"]
