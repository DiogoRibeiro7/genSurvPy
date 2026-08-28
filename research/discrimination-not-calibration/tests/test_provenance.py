"""The experiment lock, and detecting a run that does not belong to it.

The point of the lock is not to record metadata; it is to make a mismatch
*impossible to miss*. The research code and the simulation engine share a
repository, so the engine can change under a study without the version string
moving. These tests check that each way that can happen is caught.
"""

from __future__ import annotations

import json

import pytest
from survival_misspec.config import (
    EstimatorConfig,
    MetricsConfig,
    ScenarioConfig,
    StudyConfig,
)
from survival_misspec.validation import (
    LockMismatch,
    capture_provenance,
    read_lock,
    verify_lock,
    write_lock,
)


def _study(master_seed: int = 1, n_replications: int = 5) -> StudyConfig:
    return StudyConfig(
        paper_id="test-paper",
        master_seed=master_seed,
        n_replications=n_replications,
        scenarios=(ScenarioConfig("s1", "cphm", 250, 0.3, 0.5, {"beta": 0.5}),),
        estimators=(EstimatorConfig("cox_ph", "cox_ph"),),
        metrics=MetricsConfig(0.8, 51, (0.5,), ("mise",)),
    )


def _write(tmp_path, study: StudyConfig):
    return write_lock(
        tmp_path / "experiment_lock.json",
        study,
        [{"scenario_id": "s1", "tau": 1.0}],
        protocol_version="0.1.0",
        allow_dirty_tree=True,
    )


def test_provenance_records_what_could_move_a_number(tmp_path) -> None:
    provenance = capture_provenance()

    assert provenance.gen_surv_version
    assert provenance.python_version
    assert provenance.platform
    assert "numpy" in provenance.dependencies
    assert "scikit-survival" in provenance.dependencies


def test_provenance_flags_a_stale_editable_install() -> None:
    """The installed version and the declared version must agree.

    They came apart in this repository: an editable install kept serving the
    current source while its metadata stayed at an older version, so the study
    would have run 3.1.0 code and recorded 2.0.1. The flag exists so that is a
    failure rather than a mislabelled result.
    """
    provenance = capture_provenance()

    assert provenance.pyproject_version, "could not read the declared version"
    assert not provenance.version_metadata_stale, (
        f"installed gen_surv is {provenance.gen_surv_version} but pyproject.toml "
        f"declares {provenance.pyproject_version}; run `poetry install`"
    )


def test_lock_round_trips(tmp_path) -> None:
    study = _study()
    lock = _write(tmp_path, study)
    loaded = read_lock(tmp_path / "experiment_lock.json")

    assert loaded["study_hash"] == study.hash
    assert loaded["paper_id"] == "test-paper"
    assert loaded["lock_hash"] == lock.lock_hash
    assert loaded["provenance"]["git_commit"]


def test_an_unchanged_study_verifies(tmp_path) -> None:
    study = _study()
    _write(tmp_path, study)

    problems = verify_lock(
        tmp_path / "experiment_lock.json", study, strict_commit=False
    )
    assert problems == []


def test_a_changed_design_is_detected(tmp_path) -> None:
    """The case that matters most: the same code, a quietly different design."""
    _write(tmp_path, _study())

    changed = _study(n_replications=50)
    problems = verify_lock(
        tmp_path / "experiment_lock.json", changed, strict_commit=False
    )

    assert any("design changed" in problem for problem in problems)


def test_a_changed_master_seed_is_detected(tmp_path) -> None:
    _write(tmp_path, _study())

    problems = verify_lock(
        tmp_path / "experiment_lock.json", _study(master_seed=999), strict_commit=False
    )
    assert any("master seed changed" in problem for problem in problems)


def test_a_different_commit_is_detected_under_strict_verification(tmp_path) -> None:
    """gen_surv lives in this repository, so another commit may be another engine."""
    study = _study()
    path = tmp_path / "experiment_lock.json"
    _write(tmp_path, study)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["git_commit"] = "0" * 40
    path.write_text(json.dumps(payload), encoding="utf-8")

    problems = verify_lock(path, study, strict_commit=True)
    assert any("git commit changed" in problem for problem in problems)


def test_a_changed_dependency_version_is_detected(tmp_path) -> None:
    study = _study()
    path = tmp_path / "experiment_lock.json"
    _write(tmp_path, study)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["dependencies"]["numpy"] = "0.0.1-not-real"
    path.write_text(json.dumps(payload), encoding="utf-8")

    problems = verify_lock(path, study, strict_commit=False)
    assert any("numpy" in problem for problem in problems)


def test_freezing_from_a_dirty_tree_is_refused(tmp_path) -> None:
    """A lock naming a commit that lacks the code that ran is worse than none."""
    provenance = capture_provenance()
    if provenance.git_tree_clean:
        pytest.skip("working tree is clean; nothing to refuse")

    with pytest.raises(LockMismatch, match="dirty working tree"):
        write_lock(
            tmp_path / "lock.json",
            _study(),
            [],
            protocol_version="0.1.0",
            allow_dirty_tree=False,
        )


def test_lock_hash_distinguishes_experiments(tmp_path) -> None:
    first = _write(tmp_path / "a", _study())
    second = _write(tmp_path / "b", _study(master_seed=2))
    assert first.lock_hash != second.lock_hash
