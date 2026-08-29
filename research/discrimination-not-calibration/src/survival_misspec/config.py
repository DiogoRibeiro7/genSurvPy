"""Configuration schemas, and the hashes that tie a result to the design that produced it.

Every scenario, estimator and metric definition is declared in YAML under
``config/`` and parsed into a frozen dataclass here. Two rules shape the
design:

1. **A result must name the configuration that produced it.** Each object
   exposes a stable content hash, computed from a canonical JSON form so it
   does not move when keys are reordered or a float is written differently.
   Those hashes go into the experiment lock and into every result row.
2. **A configuration must fail loudly, early.** Validation happens at parse
   time, before any simulation runs, because a scenario that is only wrong at
   replication 700 has already cost hours.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .truth import SUPPORTED_DGPS, unsupported_reason

__all__ = [
    "ScenarioConfig",
    "EstimatorConfig",
    "MetricsConfig",
    "StudyConfig",
    "content_hash",
    "load_study",
]


def content_hash(payload: Any) -> str:
    """A stable SHA-256 over a canonical JSON rendering.

    ``sort_keys`` makes the hash independent of declaration order, and
    ``repr``-style float formatting keeps it stable across platforms. Truncated
    to 16 hex characters: long enough that a collision is not a practical
    concern here, short enough to read in a filename.
    """
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ScenarioConfig:
    """One data-generating mechanism at one point in the design space."""

    scenario_id: str
    dgp: str
    n: int
    target_censoring: float
    effect_size: float
    params: Mapping[str, Any]
    #: Free-text label for the *kind* of misspecification this scenario poses
    #: to a proportional-hazards model. Recorded so the paper can group
    #: scenarios by mechanism rather than by generator name.
    misspecification: str = "none"

    def __post_init__(self) -> None:
        reason = unsupported_reason(self.dgp)
        if reason is not None:
            raise ValueError(
                f"scenario {self.scenario_id!r} uses dgp {self.dgp!r}, which this "
                f"study does not support: {reason}"
            )
        if self.n <= 0:
            raise ValueError(f"scenario {self.scenario_id!r}: n must be positive")
        if not 0.0 <= self.target_censoring < 1.0:
            raise ValueError(
                f"scenario {self.scenario_id!r}: target_censoring must be in [0, 1)"
            )

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class EstimatorConfig:
    """One estimator, named by the adapter that knows how to fit it."""

    estimator_id: str
    adapter: str
    params: Mapping[str, Any] = field(default_factory=dict)
    #: What this estimator assumes, for Table 2 of the paper. Recorded here so
    #: the table is generated rather than written by hand.
    assumptions: str = ""

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class MetricsConfig:
    """The evaluation grid, fixed in advance.

    ``tau`` is the integration horizon for the truth-based losses and must be
    prespecified: choosing it after seeing results is how a null finding
    becomes a positive one. It is expressed as a quantile of the *true* marginal
    event-time distribution so that it means the same thing across scenarios
    whose time scales differ by orders of magnitude, and the realised value is
    recorded per scenario.
    """

    tau_quantile: float
    n_time_points: int
    time_grid_quantiles: Sequence[float]
    metrics: Sequence[str]

    def __post_init__(self) -> None:
        if not 0.0 < self.tau_quantile < 1.0:
            raise ValueError("tau_quantile must be in (0, 1)")
        if self.n_time_points < 2:
            raise ValueError("need at least two time points to integrate")
        if not self.metrics:
            raise ValueError("no metrics requested")

    @property
    def hash(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class StudyConfig:
    """Everything the production run needs, and nothing it does not."""

    paper_id: str
    master_seed: int
    n_replications: int
    scenarios: tuple[ScenarioConfig, ...]
    estimators: tuple[EstimatorConfig, ...]
    metrics: MetricsConfig

    def __post_init__(self) -> None:
        ids = [s.scenario_id for s in self.scenarios]
        if len(ids) != len(set(ids)):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate scenario_id: {duplicates}")

        estimator_ids = [e.estimator_id for e in self.estimators]
        if len(estimator_ids) != len(set(estimator_ids)):
            duplicates = sorted(
                {i for i in estimator_ids if estimator_ids.count(i) > 1}
            )
            raise ValueError(f"duplicate estimator_id: {duplicates}")

        if self.n_replications <= 0:
            raise ValueError("n_replications must be positive")

    @property
    def hash(self) -> str:
        """Identifies the whole design. Changing any part changes this."""
        return content_hash(
            {
                "paper_id": self.paper_id,
                "master_seed": self.master_seed,
                "n_replications": self.n_replications,
                "scenarios": sorted(s.hash for s in self.scenarios),
                "estimators": sorted(e.hash for e in self.estimators),
                "metrics": self.metrics.hash,
            }
        )

    @property
    def n_cells(self) -> int:
        return len(self.scenarios) * len(self.estimators) * self.n_replications


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return loaded


def load_study(config_dir: Path | str) -> StudyConfig:
    """Parse ``simulation.yaml``, ``estimators.yaml`` and ``metrics.yaml``."""
    directory = Path(config_dir)

    simulation = _read_yaml(directory / "simulation.yaml")
    estimators = _read_yaml(directory / "estimators.yaml")
    metrics = _read_yaml(directory / "metrics.yaml")

    scenarios = tuple(
        ScenarioConfig(
            scenario_id=entry["scenario_id"],
            dgp=entry["dgp"],
            n=int(entry["n"]),
            target_censoring=float(entry["target_censoring"]),
            effect_size=float(entry["effect_size"]),
            params=dict(entry.get("params", {})),
            misspecification=entry.get("misspecification", "none"),
        )
        for entry in simulation["scenarios"]
    )

    estimator_configs = tuple(
        EstimatorConfig(
            estimator_id=entry["estimator_id"],
            adapter=entry["adapter"],
            params=dict(entry.get("params", {})),
            assumptions=entry.get("assumptions", ""),
        )
        for entry in estimators["estimators"]
    )

    metrics_config = MetricsConfig(
        tau_quantile=float(metrics["tau_quantile"]),
        n_time_points=int(metrics["n_time_points"]),
        time_grid_quantiles=tuple(metrics["time_grid_quantiles"]),
        metrics=tuple(metrics["metrics"]),
    )

    return StudyConfig(
        paper_id=simulation["paper_id"],
        master_seed=int(simulation["master_seed"]),
        n_replications=int(simulation["n_replications"]),
        scenarios=scenarios,
        estimators=estimator_configs,
        metrics=metrics_config,
    )


def describe_supported_dgps() -> str:
    """For error messages and the protocol document."""
    return ", ".join(SUPPORTED_DGPS)
