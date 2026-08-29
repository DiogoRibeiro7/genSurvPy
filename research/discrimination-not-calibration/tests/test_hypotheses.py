"""Executable H1--H4 definitions."""

from __future__ import annotations

import pandas as pd
import pytest
from survival_misspec.hypotheses import analyse_hypotheses


def _summary() -> pd.DataFrame:
    rows = []
    for dgp, misspecification in (
        ("cphm", "none: proportional hazards, exponential baseline"),
        ("piecewise_exponential", "baseline shape only: PH holds"),
        ("aft_ln", "non-proportional hazards"),
        ("aft_log_logistic", "non-proportional hazards"),
        ("mixture_cure", "survival plateau"),
    ):
        censoring_levels = [0.5, 0.7] if dgp == "mixture_cure" else [0.1, 0.7]
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
    hypotheses = analyse_hypotheses(_summary())

    assert set(hypotheses["hypothesis"]) == {"H1", "H2", "H3", "H4"}
    assert hypotheses["estimand"].notna().all()
    assert hypotheses["criterion"].notna().all()


def test_h2_uses_only_dgps_with_mapped_correct_references() -> None:
    h2 = analyse_hypotheses(_summary()).set_index("hypothesis").loc["H2"]

    assert h2["estimate"] == pytest.approx(400.0)
    assert "aft_ln" not in h2["note"]
    assert bool(h2["supports_hypothesis"])


def test_h3_and_h4_use_common_support() -> None:
    hypotheses = analyse_hypotheses(_summary()).set_index("hypothesis")

    assert hypotheses.loc["H3", "n"] == 4
    assert hypotheses.loc["H4", "n"] == 12
    assert "common_support" in hypotheses.loc["H3", "estimand"]
    assert "common_support" in hypotheses.loc["H4", "estimand"]
