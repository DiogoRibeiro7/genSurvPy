"""Shared metadata and validation helpers for pre-freeze gate artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from survival_misspec.config import StudyConfig, content_hash
from survival_misspec.validation import capture_provenance


def file_sha256(path: Path) -> str:
    """SHA-256 digest of an artifact exactly as written."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_design_hash(study: StudyConfig) -> str:
    """Hash of the declared scenario set independent of row order."""
    return content_hash(
        {scenario.scenario_id: scenario.hash for scenario in study.scenarios}
    )


def estimator_design_hash(study: StudyConfig) -> str:
    """Hash of the declared estimator set independent of row order."""
    return content_hash(
        {estimator.estimator_id: estimator.hash for estimator in study.estimators}
    )


def study_metadata(study: StudyConfig) -> dict[str, object]:
    """Metadata every pre-freeze gate artifact must carry."""
    provenance = capture_provenance()
    return {
        "study_hash": study.hash,
        "git_commit": provenance.git_commit,
        "git_tree_clean": provenance.git_tree_clean,
        "n_replications_planned": study.n_replications,
        "scenario_design_hash": scenario_design_hash(study),
        "estimator_design_hash": estimator_design_hash(study),
        "metrics_hash": study.metrics.hash,
    }


def add_metadata(
    row: dict[str, object], metadata: dict[str, object]
) -> dict[str, object]:
    """Return a row with common gate metadata appended."""
    return {**row, **metadata}


def _unique(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame.columns:
        return []
    return frame[column].drop_duplicates().tolist()


def metadata_problems(
    frame: pd.DataFrame,
    study: StudyConfig,
    *,
    expected: dict[str, object] | None = None,
) -> list[str]:
    """Validate gate metadata against the current declared study."""
    required = {
        "study_hash",
        "git_commit",
        "git_tree_clean",
        "n_replications_planned",
        "scenario_design_hash",
        "estimator_design_hash",
        "metrics_hash",
    }
    problems: list[str] = []
    missing = sorted(required - set(frame.columns))
    if missing:
        return [f"gate artifact missing metadata columns: {missing}"]

    expected_values: dict[str, object] = {
        "study_hash": study.hash,
        "n_replications_planned": study.n_replications,
        "scenario_design_hash": scenario_design_hash(study),
        "estimator_design_hash": estimator_design_hash(study),
        "metrics_hash": study.metrics.hash,
    }
    if expected:
        expected_values.update(expected)

    for column, wanted in expected_values.items():
        values = _unique(frame, column)
        if len(values) != 1 or values[0] != wanted:
            problems.append(
                f"{column} mismatch: artifact has {values[:3]} but expected {wanted}"
            )

    clean_values = _unique(frame, "git_tree_clean")
    if clean_values != [True]:
        problems.append("gate artifact was not produced from a clean tree")

    current_commit = capture_provenance().git_commit
    commit_values = _unique(frame, "git_commit")
    if len(commit_values) != 1 or commit_values[0] != current_commit:
        problems.append(
            f"git_commit mismatch: artifact has {commit_values[:3]} "
            f"but current is {current_commit}"
        )

    return problems
