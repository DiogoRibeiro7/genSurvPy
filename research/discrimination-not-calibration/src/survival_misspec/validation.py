"""Provenance, and the experiment lock that makes a result traceable.

The publication-readiness criterion for this study is that any number in the
paper can be walked back to the commit that produced it:

    result -> replication -> seed -> scenario -> DGP parameters -> estimator
           -> metric -> gen_surv version -> git commit

This module supplies the two ends of that chain. :func:`capture_provenance`
records the environment; :func:`write_lock` freezes it together with the design
and :func:`verify_lock` refuses to let a run continue against a different one.

The awkward part is that the research code and the simulation engine live in
the same repository. Recording ``gen-surv==3.1.0`` is not enough, because the
generators can change while the version string does not. So the lock records
the **commit** and whether the working tree was clean, and every result row
carries the lock hash. A result produced from a different implementation is
then detectable rather than merely unlikely.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import StudyConfig, content_hash

__all__ = [
    "Provenance",
    "capture_provenance",
    "write_lock",
    "read_lock",
    "verify_lock",
    "LockMismatch",
]

#: Packages whose version can change a numerical result. Recorded exactly.
TRACKED_DEPENDENCIES = (
    "gen-surv",
    "numpy",
    "pandas",
    "scipy",
    "scikit-survival",
    "scikit-learn",
    "lifelines",
    "pyarrow",
)


class LockMismatch(RuntimeError):
    """Raised when a run does not match the experiment it claims to be part of."""


def _git(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], capture_output=True, text=True, check=False
        )
    except OSError:  # pragma: no cover - git absent
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _pyproject_version() -> str:
    """The version declared in the repository, as distinct from the installed one."""
    import re

    root = Path(__file__).resolve().parents[4]
    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return ""
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8")
    )
    return match.group(1) if match else ""


def _dependency_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in TRACKED_DEPENDENCIES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@dataclass(frozen=True)
class Provenance:
    """Everything about the environment that could move a number."""

    gen_surv_version: str
    pyproject_version: str
    version_metadata_stale: bool
    git_commit: str
    git_branch: str
    git_tree_clean: bool
    git_dirty_files: tuple[str, ...]
    python_version: str
    platform: str
    dependencies: Mapping[str, str]
    captured_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture_provenance() -> Provenance:
    """Record the current environment.

    ``git_tree_clean`` is the one to watch. A production run from a dirty tree
    is not reproducible from its commit, because the commit does not contain
    the code that ran. :func:`write_lock` refuses it unless explicitly allowed.
    """
    import gen_surv

    status = _git("status", "--porcelain")
    dirty = tuple(line[3:] for line in status.splitlines() if line.strip())

    # `gen_surv.__version__` reads installed metadata, not the source being
    # imported. In this repository the two come apart easily: an editable
    # install keeps serving the current source while its dist-info stays at
    # whatever version was installed. That happened here -- the engine reported
    # 2.0.1 while running 3.1.0 code -- and every provenance record would have
    # named the wrong version. Recorded as a flag so it fails loudly rather
    # than silently mislabelling a study; `poetry install` resolves it.
    installed = getattr(gen_surv, "__version__", "unknown")
    declared = _pyproject_version()

    return Provenance(
        gen_surv_version=installed,
        pyproject_version=declared,
        version_metadata_stale=bool(declared and installed != declared),
        git_commit=_git("rev-parse", "HEAD"),
        git_branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
        git_tree_clean=not dirty,
        git_dirty_files=dirty,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=_dependency_versions(),
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


@dataclass(frozen=True)
class ExperimentLock:
    """The frozen definition of one experiment version."""

    paper_id: str
    protocol_version: str
    study_hash: str
    master_seed: int
    n_replications: int
    scenarios: tuple[Mapping[str, Any], ...]
    estimators: tuple[Mapping[str, Any], ...]
    metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    gate_evidence: Mapping[str, Any]
    frozen_at: str
    notes: str = ""
    _lock_hash: str = field(default="", compare=False)

    @property
    def lock_hash(self) -> str:
        """Identity of the experiment. Every result row carries it."""
        payload = {
            "paper_id": self.paper_id,
            "protocol_version": self.protocol_version,
            "study_hash": self.study_hash,
            "master_seed": self.master_seed,
            "n_replications": self.n_replications,
            "scenarios": self.scenarios,
            "estimators": self.estimators,
            "metrics": self.metrics,
            "gate_evidence": self.gate_evidence,
            "git_commit": self.provenance.get("git_commit"),
            "git_tree_clean": self.provenance.get("git_tree_clean"),
            "gen_surv_version": self.provenance.get("gen_surv_version"),
            "python_version": self.provenance.get("python_version"),
            "platform": self.provenance.get("platform"),
            "dependencies": self.provenance.get("dependencies", {}),
        }
        return content_hash(payload)


def write_lock(
    path: Path | str,
    study: StudyConfig,
    prepared_scenarios: list[Mapping[str, Any]],
    protocol_version: str,
    *,
    notes: str = "",
    allow_dirty_tree: bool = False,
    gate_evidence: Mapping[str, Any] | None = None,
) -> ExperimentLock:
    """Freeze the experiment. Refuses a dirty tree unless told otherwise.

    A lock written from an uncommitted working tree names a commit that does
    not contain the code that ran, which defeats the purpose. The override
    exists for pilots and dry runs, which are not production results.
    """
    provenance = capture_provenance()
    if provenance.version_metadata_stale:
        raise LockMismatch(
            f"installed gen_surv metadata says {provenance.gen_surv_version} but "
            f"pyproject.toml declares {provenance.pyproject_version}. The source "
            f"being imported is the repository's, so the study would run "
            f"{provenance.pyproject_version} code while recording "
            f"{provenance.gen_surv_version}. Run `poetry install` first."
        )
    if not provenance.git_tree_clean and not allow_dirty_tree:
        raise LockMismatch(
            "refusing to freeze an experiment from a dirty working tree; the "
            "recorded commit would not contain the code that ran. Uncommitted:\n  "
            + "\n  ".join(provenance.git_dirty_files[:20])
            + "\n\nCommit first, or pass allow_dirty_tree=True for a pilot."
        )

    lock = ExperimentLock(
        paper_id=study.paper_id,
        protocol_version=protocol_version,
        study_hash=study.hash,
        master_seed=study.master_seed,
        n_replications=study.n_replications,
        scenarios=tuple(prepared_scenarios),
        estimators=tuple(asdict(e) for e in study.estimators),
        metrics=asdict(study.metrics),
        provenance=provenance.as_dict(),
        gate_evidence=gate_evidence or {},
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )

    payload = asdict(lock)
    payload.pop("_lock_hash", None)
    payload["lock_hash"] = lock.lock_hash

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return lock


def read_lock(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _expected_lock_hash(lock: Mapping[str, Any]) -> str:
    payload = {
        "paper_id": lock.get("paper_id"),
        "protocol_version": lock.get("protocol_version"),
        "study_hash": lock.get("study_hash"),
        "master_seed": lock.get("master_seed"),
        "n_replications": lock.get("n_replications"),
        "scenarios": lock.get("scenarios", ()),
        "estimators": lock.get("estimators", ()),
        "metrics": lock.get("metrics", {}),
        "gate_evidence": lock.get("gate_evidence", {}),
        "git_commit": lock.get("provenance", {}).get("git_commit"),
        "git_tree_clean": lock.get("provenance", {}).get("git_tree_clean"),
        "gen_surv_version": lock.get("provenance", {}).get("gen_surv_version"),
        "python_version": lock.get("provenance", {}).get("python_version"),
        "platform": lock.get("provenance", {}).get("platform"),
        "dependencies": lock.get("provenance", {}).get("dependencies", {}),
    }
    return content_hash(payload)


def verify_lock(
    path: Path | str, study: StudyConfig, *, strict_commit: bool = True
) -> list[str]:
    """Return the reasons the current state does not match the lock.

    An empty list means the run may proceed. Every entry names one difference,
    so a mismatch is diagnosable rather than a bare refusal.

    ``strict_commit`` is the difference between a production run and an
    exploratory one. In production a different commit means a different
    experiment, full stop -- the spec's rule that package code changing after
    the freeze starts a new experimental version rather than silently
    continuing the old one.
    """
    lock = read_lock(path)
    current = capture_provenance()
    problems: list[str] = []

    stored_hash = lock.get("lock_hash")
    expected_hash = _expected_lock_hash(lock)
    if stored_hash != expected_hash:
        problems.append(
            f"lock hash changed: lock records {stored_hash or '<missing>'} "
            f"but its contents hash to {expected_hash}"
        )

    if lock.get("study_hash") != study.hash:
        problems.append(
            f"design changed: lock study_hash={lock.get('study_hash')} "
            f"but current={study.hash}. The scenarios, estimators, metrics, "
            f"seed or replication count differ from the frozen experiment."
        )

    if lock.get("master_seed") != study.master_seed:
        problems.append(
            f"master seed changed: {lock.get('master_seed')} -> {study.master_seed}"
        )

    locked_scenarios = lock.get("scenarios", [])
    if not isinstance(locked_scenarios, list) or not locked_scenarios:
        problems.append("lock does not contain prepared scenarios")
    else:
        current_hashes = {
            scenario.scenario_id: scenario.hash for scenario in study.scenarios
        }
        locked_hashes = {
            str(record.get("scenario_id")): record.get("scenario_hash")
            for record in locked_scenarios
            if isinstance(record, Mapping)
        }
        missing = sorted(set(current_hashes) - set(locked_hashes))
        extra = sorted(set(locked_hashes) - set(current_hashes))
        if missing:
            problems.append(f"lock is missing prepared scenarios: {missing[:10]}")
        if extra:
            problems.append(f"lock contains unknown prepared scenarios: {extra[:10]}")
        for scenario_id, current_hash in current_hashes.items():
            locked_hash = locked_hashes.get(scenario_id)
            if locked_hash is not None and locked_hash != current_hash:
                problems.append(
                    f"scenario {scenario_id} changed: lock scenario_hash="
                    f"{locked_hash} but current={current_hash}"
                )

        required_fields = {"params", "tau", "time_grid", "ipcw_time_grid", "feasible"}
        for record in locked_scenarios:
            if not isinstance(record, Mapping):
                problems.append("lock contains a non-mapping prepared scenario")
                continue
            missing_fields = sorted(required_fields - set(record))
            if missing_fields:
                problems.append(
                    f"prepared scenario {record.get('scenario_id', '?')} is missing "
                    f"{missing_fields}"
                )

    locked_provenance = lock.get("provenance", {})

    if strict_commit and locked_provenance.get("git_commit") != current.git_commit:
        problems.append(
            f"git commit changed: experiment was frozen at "
            f"{locked_provenance.get('git_commit', '?')[:12]} but this is "
            f"{current.git_commit[:12]}. gen_surv itself lives in this "
            f"repository, so a different commit may be a different simulation "
            f"engine. Treat this as a new experiment version."
        )

    if strict_commit and not locked_provenance.get("git_tree_clean", False):
        problems.append("experiment lock was created from a dirty working tree")

    if strict_commit and not current.git_tree_clean:
        problems.append(
            "working tree is dirty, so this run is not reproducible from any "
            "commit: " + ", ".join(current.git_dirty_files[:10])
        )

    if current.version_metadata_stale:
        problems.append(
            f"installed gen_surv metadata ({current.gen_surv_version}) disagrees "
            f"with pyproject.toml ({current.pyproject_version}); run "
            f"`poetry install` so results are labelled with the code that ran"
        )

    locked_deps = locked_provenance.get("dependencies", {})
    for name, locked_version in locked_deps.items():
        actual = current.dependencies.get(name, "not installed")
        if actual != locked_version:
            problems.append(f"{name}: locked {locked_version}, found {actual}")

    if locked_provenance.get("python_version") != current.python_version:
        problems.append(
            f"python: locked {locked_provenance.get('python_version', '?')}, "
            f"found {current.python_version}"
        )

    if locked_provenance.get("platform") != current.platform:
        problems.append(
            f"platform: locked {locked_provenance.get('platform', '?')}, "
            f"found {current.platform}"
        )

    return problems
