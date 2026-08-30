"""Fast checks for the pre-freeze acceptance gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from survival_misspec.aggregation import headline_metric_gap
from survival_misspec.config import (
    EstimatorConfig,
    MetricsConfig,
    ScenarioConfig,
    StudyConfig,
    content_hash,
)
from survival_misspec.validation import capture_provenance

ROOT = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(study: StudyConfig, **extra: object) -> dict[str, object]:
    provenance = capture_provenance()
    return {
        "study_hash": study.hash,
        "git_commit": provenance.git_commit,
        "git_tree_clean": True,
        "n_replications_planned": study.n_replications,
        "scenario_design_hash": content_hash(
            {scenario.scenario_id: scenario.hash for scenario in study.scenarios}
        ),
        "estimator_design_hash": content_hash(
            {estimator.estimator_id: estimator.hash for estimator in study.estimators}
        ),
        "metrics_hash": study.metrics.hash,
        **extra,
    }


def test_headline_bootstrap_is_deterministic_under_fixed_seed() -> None:
    summary = pd.DataFrame(
        {
            "c_index_harrell_mean": [0.60, 0.61, 0.70, 0.71],
            "c_index_harrell_integrated_hazard_mean": [0.60, 0.61, 0.70, 0.71],
            "root_mean_integrated_squared_error_mean": [0.05, 0.20, 0.04, 0.30],
            "root_mean_integrated_squared_error_mcse": [0.002, 0.003, 0.002, 0.004],
        }
    )

    first = headline_metric_gap(
        summary, bins=2, quantile=0.9, uncertainty_draws=200, seed=12
    )
    second = headline_metric_gap(
        summary, bins=2, quantile=0.9, uncertainty_draws=200, seed=12
    )

    pd.testing.assert_frame_equal(first, second)
    assert "loss_quantile_se" in first.columns
    assert "bootstrap_mc_error" in first.columns
    assert "loss_quantile_mcse" not in first.columns


def test_headline_bootstrap_se_is_not_divided_by_number_of_draws() -> None:
    summary = pd.DataFrame(
        {
            "c_index_harrell_mean": [0.60, 0.61, 0.70, 0.71],
            "c_index_harrell_integrated_hazard_mean": [0.60, 0.61, 0.70, 0.71],
            "root_mean_integrated_squared_error_mean": [0.05, 0.20, 0.04, 0.30],
            "root_mean_integrated_squared_error_mcse": [0.002, 0.003, 0.002, 0.004],
        }
    )

    headline = headline_metric_gap(
        summary, bins=2, quantile=0.9, uncertainty_draws=200, seed=12
    )

    assert (headline["loss_quantile_se"] > headline["bootstrap_mc_error"]).all()


def test_ipcw_availability_criterion_ignores_infeasible_scenarios() -> None:
    audit = _load_script("audit_ipcw_availability.py")
    frame = pd.DataFrame(
        {
            "scenario_id": ["ok", "bad", "missing", "infeasible"],
            "feasible": [True, True, True, False],
            "availability": [0.95, 0.94, np.nan, np.nan],
        }
    )

    assert not audit.availability_passes(frame, minimum_availability=0.95)
    failures = audit.availability_failures(frame, minimum_availability=0.95)
    assert failures["scenario_id"].tolist() == ["bad", "missing"]


def test_ipcw_support_requires_the_whole_grid_inside_followup_support() -> None:
    audit = _load_script("audit_ipcw_availability.py")

    assert audit._supported(
        np.array([0.2, 0.5]), np.array([0.1, 0.8]), np.array([0.0, 0.7])
    )
    assert not audit._supported(
        np.array([0.2, 0.9]), np.array([0.1, 0.8]), np.array([0.0, 0.7])
    )


def test_ipcw_audit_uses_production_evaluation_size(monkeypatch) -> None:
    audit = _load_script("audit_ipcw_availability.py")
    calls = []

    def fake_draw_replicate(dgp, params, n, scenario_id, replication_id, seed, stream):
        calls.append((stream, n))
        return SimpleNamespace(observed_time=np.array([0.0, 2.0]))

    monkeypatch.setattr(audit, "draw_replicate", fake_draw_replicate)
    scenario = SimpleNamespace(dgp="cphm", n=250, scenario_id="s1")
    prepared = SimpleNamespace(params={}, ipcw_time_grid=(0.5, 1.0))

    supported, availability = audit.estimate_availability(
        scenario, prepared, master_seed=123, replications=3
    )

    assert supported == 3
    assert availability == 1.0
    assert calls == [
        ("train", 250),
        ("eval", audit.EVALUATION_N),
        ("train", 250),
        ("eval", audit.EVALUATION_N),
        ("train", 250),
        ("eval", audit.EVALUATION_N),
    ]


def test_grid_convergence_pass_fail_uses_preregistered_rmise_tolerance() -> None:
    grid = _load_script("check_grid_convergence.py")
    frame = pd.DataFrame(
        {
            "n_time_points": [51, 801],
            "rmise_absolute_difference": [0.0021, 0.0],
            "c_index_harrell_integrated_hazard_absolute_difference": [0.001, 0.0],
        }
    )

    assert not grid.grid_convergence_passes(
        frame, reference_grid=801, rmise_epsilon=0.002
    )
    assert grid.grid_convergence_passes(frame, reference_grid=801, rmise_epsilon=0.003)
    assert grid.maximum_rmise_difference(frame, 801) == pytest.approx(0.0021)
    assert grid.maximum_c_index_difference(frame, 801) == pytest.approx(0.001)


def test_grid_convergence_selects_worst_cells_from_summary() -> None:
    grid = _load_script("check_grid_convergence.py")
    study = StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=1,
        scenarios=(
            ScenarioConfig("s1", "cphm", 100, 0.3, 0.5, {}),
            ScenarioConfig("s2", "cphm", 100, 0.3, 0.5, {}),
        ),
        estimators=(
            EstimatorConfig("cox_ph", "cox_ph"),
            EstimatorConfig("weibull_aft", "weibull_aft"),
        ),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )
    summary = pd.DataFrame(
        {
            "scenario_id": ["s1", "s1", "s2", "outside"],
            "estimator_id": ["cox_ph", "weibull_aft", "cox_ph", "cox_ph"],
            "root_mean_integrated_squared_error_mean": [0.1, 0.4, 0.2, 99.0],
            "mise_mean": [0.5, 0.1, 0.9, 100.0],
        }
    )

    selected = grid.select_audit_cells(study, summary=summary, top_cells=2)

    assert selected == {("s1", "weibull_aft"), ("s2", "cox_ph")}


def test_grid_convergence_loads_frozen_audit_cells(tmp_path) -> None:
    grid = _load_script("check_grid_convergence.py")
    study = StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=1,
        scenarios=(ScenarioConfig("s1", "cphm", 100, 0.3, 0.5, {}),),
        estimators=(EstimatorConfig("cox_ph", "cox_ph"),),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )
    path = tmp_path / "grid_audit_cells.json"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "scenario_id": "s1",
                        "scenario_hash": study.scenarios[0].hash,
                        "estimator_id": "cox_ph",
                        "estimator_hash": study.estimators[0].hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    selected = grid.load_audit_cells(study, path, expected_count=1)

    assert selected == {("s1", "cox_ph")}


def test_grid_convergence_rejects_stale_frozen_audit_cells(tmp_path) -> None:
    grid = _load_script("check_grid_convergence.py")
    study = StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=1,
        scenarios=(ScenarioConfig("s1", "cphm", 100, 0.3, 0.5, {}),),
        estimators=(EstimatorConfig("cox_ph", "cox_ph"),),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )
    path = tmp_path / "grid_audit_cells.json"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "scenario_id": "s1",
                        "scenario_hash": "stale",
                        "estimator_id": "cox_ph",
                        "estimator_hash": study.estimators[0].hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scenario_hash mismatch"):
        grid.load_audit_cells(study, path, expected_count=1)


def test_grid_convergence_requires_exact_frozen_cell_count(tmp_path) -> None:
    grid = _load_script("check_grid_convergence.py")
    study = StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=1,
        scenarios=(ScenarioConfig("s1", "cphm", 100, 0.3, 0.5, {}),),
        estimators=(EstimatorConfig("cox_ph", "cox_ph"),),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )
    path = tmp_path / "grid_audit_cells.json"
    path.write_text(
        json.dumps({"cells": [{"scenario_id": "s1", "estimator_id": "cox_ph"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contains 1 cells; expected 2"):
        grid.load_audit_cells(study, path, expected_count=2)


def test_freeze_gate_evidence_records_passed_artifact_hashes(tmp_path) -> None:
    freeze = _load_script("freeze_experiment.py")
    ipcw = tmp_path / "ipcw.parquet"
    grid = tmp_path / "grid.parquet"
    audit_cells = tmp_path / "grid_audit_cells.json"
    study = StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=10,
        scenarios=(
            ScenarioConfig("s1", "cphm", 100, 0.3, 0.5, {}),
            ScenarioConfig("s2", "cphm", 100, 0.3, 0.5, {}),
        ),
        estimators=(EstimatorConfig("cox_ph", "cox_ph"),),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )
    ipcw_metadata = _metadata(
        study,
        audited_replications=10,
        minimum_availability_threshold=0.95,
    )
    pd.DataFrame(
        [
            {
                "scenario_id": "s1",
                "scenario_hash": study.scenarios[0].hash,
                "feasible": True,
                "availability": 0.96,
                "attempted": 10,
                **ipcw_metadata,
            },
            {
                "scenario_id": "s2",
                "scenario_hash": study.scenarios[1].hash,
                "feasible": False,
                "availability": np.nan,
                "attempted": 0,
                **ipcw_metadata,
            },
        ]
    ).to_parquet(ipcw, index=False)
    audit_cells.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "scenario_id": "s1",
                        "scenario_hash": study.scenarios[0].hash,
                        "estimator_id": "cox_ph",
                        "estimator_hash": study.estimators[0].hash,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    grid_metadata = _metadata(
        study,
        audited_replications=10,
        rmise_epsilon=0.002,
        c_index_epsilon=0.002,
        audit_cells_sha256=freeze.file_sha256(audit_cells),
        top_cells=1,
        loss_column="",
    )
    grid_rows = []
    for replication_id in range(10):
        for points, rmise_difference in ((51, 0.001), (801, 0.0)):
            grid_rows.append(
                {
                    "scenario_id": "s1",
                    "scenario_hash": study.scenarios[0].hash,
                    "estimator_id": "cox_ph",
                    "estimator_hash": study.estimators[0].hash,
                    "replication_id": replication_id,
                    "n_time_points": points,
                    "reference_n_time_points": 801,
                    "rmise_absolute_difference": rmise_difference,
                    "c_index_harrell_integrated_hazard_absolute_difference": (
                        0.001 if points == 51 else 0.0
                    ),
                    **grid_metadata,
                }
            )
    pd.DataFrame(grid_rows).to_parquet(grid, index=False)

    ipcw_evidence = freeze._ipcw_gate_evidence(ipcw, 0.95, study)
    grid_evidence = freeze._grid_gate_evidence(
        grid, 0.002, 0.002, study, audit_cells, 1, 10
    )

    assert ipcw_evidence["status"] == "PASS"
    assert ipcw_evidence["minimum_availability"] == pytest.approx(0.96)
    assert len(ipcw_evidence["sha256"]) == 64
    assert grid_evidence["status"] == "PASS"
    assert grid_evidence["maximum_rmise_difference"] == pytest.approx(0.001)
    assert grid_evidence["maximum_c_index_difference"] == pytest.approx(0.001)
    assert len(grid_evidence["sha256"]) == 64


def test_hypothesis_analysis_writes_manuscript_macros(tmp_path) -> None:
    analyze = _load_script("analyze_hypotheses.py")
    hypotheses = pd.DataFrame(
        {
            "hypothesis": ["H1", "H2"],
            "estimate": [0.123456, 12.0],
            "estimate_se": [0.01, 0.5],
            "estimate_ci_low": [0.10, 11.0],
            "estimate_ci_high": [0.15, 13.0],
            "n": [30, 4],
            "supports_hypothesis": [True, False],
        }
    )
    out = tmp_path / "hypotheses.tex"

    analyze._write_macros(hypotheses, out, tmp_path / "hypotheses.parquet")

    text = out.read_text(encoding="utf-8")
    assert "Generated by scripts/analyze_hypotheses.py" in text
    assert r"\newcommand{\HypothesisHOneEstimate}{0.1235}" in text
    assert r"\newcommand{\HypothesisHOneSE}{0.0100}" in text
    assert r"\newcommand{\HypothesisHOneCILow}{0.1000}" in text
    assert r"\newcommand{\HypothesisHOneCIHigh}{0.1500}" in text
    assert r"\newcommand{\HypothesisHTwoDecision}{does not support}" in text
