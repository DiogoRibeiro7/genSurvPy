"""Artifact failures identify the file while preserving the underlying cause."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import yaml
from dataexcept import DataLoadingError, FileWriteError
from survival_misspec import validation
from survival_misspec.aggregation import read_raw, write_raw
from survival_misspec.config import (
    EstimatorConfig,
    MetricsConfig,
    ScenarioConfig,
    StudyConfig,
    load_study,
)
from survival_misspec.validation import read_lock, write_lock


def test_missing_study_configuration_identifies_file(tmp_path) -> None:
    path = tmp_path / "simulation.yaml"

    with pytest.raises(DataLoadingError, match="simulation.yaml") as error:
        load_study(tmp_path)

    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert str(path) in str(error.value)


def test_malformed_study_configuration_preserves_yaml_error(tmp_path) -> None:
    (tmp_path / "simulation.yaml").write_text(
        "scenarios: [unfinished", encoding="utf-8"
    )

    with pytest.raises(DataLoadingError, match="simulation.yaml") as error:
        load_study(tmp_path)

    assert isinstance(error.value.__cause__, yaml.YAMLError)


def test_corrupt_raw_shard_identifies_source(tmp_path) -> None:
    path = tmp_path / "raw.parquet"
    path.write_bytes(b"broken parquet")

    with pytest.raises(DataLoadingError, match="raw.parquet") as error:
        read_raw(path)

    assert isinstance(error.value.__cause__, (OSError, ValueError))


def test_raw_shard_write_preserves_filesystem_failure(tmp_path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("regular file", encoding="utf-8")

    with pytest.raises(FileWriteError, match="blocked") as error:
        write_raw([{"scenario_id": "s1"}], blocked / "raw.parquet")

    assert isinstance(error.value.__cause__, OSError)


def test_corrupt_experiment_lock_preserves_json_error(tmp_path) -> None:
    path = tmp_path / "experiment_lock.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(DataLoadingError, match="experiment_lock.json") as error:
        read_lock(path)

    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_experiment_lock_write_preserves_filesystem_failure(
    tmp_path, monkeypatch
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("regular file", encoding="utf-8")

    # The environment's installed metadata need not match the checkout used
    # for this filesystem test; only provenance's version flag is bypassed.
    provenance = validation.capture_provenance()
    monkeypatch.setattr(
        validation,
        "capture_provenance",
        lambda: replace(provenance, version_metadata_stale=False),
    )

    study = StudyConfig(
        paper_id="test-paper",
        master_seed=1,
        n_replications=1,
        scenarios=(ScenarioConfig("s1", "cphm", 250, 0.3, 0.5, {"beta": 0.5}),),
        estimators=(EstimatorConfig("cox", "cox_ph"),),
        metrics=MetricsConfig(0.8, 51, (0.5,), ("mise",)),
    )

    with pytest.raises(FileWriteError, match="experiment_lock.json") as error:
        write_lock(
            blocked / "experiment_lock.json",
            study,
            [],
            protocol_version="0.1.0",
            allow_dirty_tree=True,
        )

    assert isinstance(error.value.__cause__, OSError)
