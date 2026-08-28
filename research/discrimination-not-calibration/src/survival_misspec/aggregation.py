"""Aggregating replicate rows, with Monte Carlo error carried throughout.

A simulation study reports estimates of expectations, and an estimate without
its own uncertainty is not a result. For a Monte Carlo mean over ``R``
independent replications,

.. math::

    \\widehat\\mu = \\frac{1}{R}\\sum_{r=1}^{R} Y_r,
    \\qquad
    \\mathrm{MCSE}(\\widehat\\mu) = \\frac{s_Y}{\\sqrt{R}},

so every aggregated column here comes with its MCSE. Two differences whose
MCSEs overlap are not distinguishable at the replication count used, and the
paper must not describe them as if they were.

Failures are aggregated too, not dropped. ``P(fit failure | scenario,
estimator)`` is reported alongside the metrics, because an estimator that fails
on a third of replicates and scores well on the rest has not done better than
one that fits every time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "mcse",
    "replications_for_precision",
    "aggregate",
    "failure_rates",
    "adequacy_region",
    "write_raw",
    "read_raw",
    "completed_cells",
]

#: Columns aggregated with a mean and an MCSE. Anything not listed is either an
#: identifier, a diagnostic string, or already a summary.
METRIC_COLUMNS = (
    "mise",
    "miae",
    "mean_absolute_survival_error",
    "miae_p90",
    "c_index_harrell",
    "c_index_uno",
    "c_index_at_tau",
    "c_index_antolini",
    "d_calibration_p",
    "d_calibration_normalised",
    "integrated_brier_score",
    "brier_at_tau",
    "auc_mean",
    "calibration_error",
    "calibration_error_max",
    "fit_runtime_seconds",
    "beta_bias_scalar",
)


def mcse(values: Sequence[float] | np.ndarray) -> float:
    """Monte Carlo standard error of the mean of ``values``."""
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    return float(np.std(array, ddof=1) / math.sqrt(array.size))


def replications_for_precision(
    pilot_values: Sequence[float] | np.ndarray, target_mcse: float
) -> int:
    """How many replications are needed for a given Monte Carlo precision.

    Inverts :math:`\\mathrm{MCSE} = s/\\sqrt{R}` using the pilot's standard
    deviation. This is how ``R`` should be chosen -- from the precision the
    conclusions require -- rather than by taking 100, 500 or 1000 because those
    are the numbers people use.
    """
    array = np.asarray(pilot_values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size < 2 or target_mcse <= 0:
        return 0
    return int(math.ceil((float(np.std(array, ddof=1)) / target_mcse) ** 2))


def _present(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def aggregate(
    raw: pd.DataFrame,
    by: Sequence[str] = ("scenario_id", "estimator_id"),
) -> pd.DataFrame:
    """Mean, MCSE and replication count per cell, over successful scorings only.

    Rows where the fit or the scoring failed carry no metric values, so they are
    excluded from the metric means -- but they are counted, and
    :func:`failure_rates` reports them separately. Aggregating over successes
    while hiding the failure count is how a fragile estimator comes to look
    strong.
    """
    group_keys = list(by)
    scored = raw[raw.get("scored", pd.Series(False, index=raw.index)).fillna(False)]

    metric_columns = _present(scored, METRIC_COLUMNS)
    records: list[dict[str, object]] = []

    for keys, block in raw.groupby(group_keys, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_keys, key_values))

        record["n_replications_attempted"] = int(len(block))
        scored_block = block[
            block.get("scored", pd.Series(False, index=block.index)).fillna(False)
        ]
        record["n_replications_scored"] = int(len(scored_block))

        for column in metric_columns:
            values = pd.to_numeric(scored_block[column], errors="coerce").to_numpy()
            finite = values[np.isfinite(values)]
            record[f"{column}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            record[f"{column}_mcse"] = mcse(finite)
            record[f"{column}_n"] = int(finite.size)

        # Carry design columns through unchanged when they are constant in the
        # group, so the aggregate table can be read without a join.
        for column in (
            "dgp",
            "n",
            "target_censoring",
            "effect_size",
            "misspecification",
            "tau",
            "adapter",
        ):
            if column in block.columns and block[column].nunique(dropna=False) == 1:
                record[column] = block[column].iloc[0]

        records.append(record)

    return pd.DataFrame.from_records(records)


def failure_rates(
    raw: pd.DataFrame, by: Sequence[str] = ("scenario_id", "estimator_id")
) -> pd.DataFrame:
    """``P(fit failure)`` and ``P(scoring failure)`` per cell, with the reasons."""
    group_keys = list(by)
    records: list[dict[str, object]] = []

    for keys, block in raw.groupby(group_keys, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_keys, key_values))

        attempted = len(block)
        fit_failures = int((~block["fitted"].fillna(False)).sum())
        scored = block.get("scored", pd.Series(False, index=block.index)).fillna(False)
        score_failures = int((block["fitted"].fillna(False) & ~scored).sum())

        record.update(
            {
                "attempted": attempted,
                "fit_failures": fit_failures,
                "fit_failure_rate": (
                    fit_failures / attempted if attempted else float("nan")
                ),
                "score_failures": score_failures,
                "score_failure_rate": (
                    score_failures / attempted if attempted else float("nan")
                ),
                "fit_error_types": "; ".join(
                    sorted(
                        {
                            e
                            for e in block.get("fit_error_type", [])
                            if isinstance(e, str) and e
                        }
                    )
                )[:300],
                "warning_types": "; ".join(
                    sorted(
                        {
                            w.split(":")[0]
                            for w in block.get("fit_warnings", [])
                            if isinstance(w, str) and w
                        }
                    )
                )[:300],
            }
        )
        records.append(record)

    return pd.DataFrame.from_records(records)


def adequacy_region(
    aggregated: pd.DataFrame,
    reference_estimator: str,
    loss: str = "mise_mean",
    epsilon: float = 0.01,
    by: Sequence[str] = ("scenario_id",),
) -> pd.DataFrame:
    r"""Where a candidate's loss stays within ``epsilon`` of the reference's.

    .. math::

        \mathcal{R}_\epsilon = \{\theta :
        L_{\text{candidate}}(\theta) - L_{\text{reference}}(\theta) \le \epsilon\}

    ``epsilon`` is **not** a universal threshold. It is a quantity on the scale
    of the chosen loss, meaningful only relative to this DGP, this horizon and
    an application's tolerance for absolute probability error. The function
    takes it as an argument for exactly that reason: the paper reports the
    region as a function of ``epsilon`` rather than asserting one value.
    """
    keys = list(by)
    reference = aggregated[aggregated["estimator_id"] == reference_estimator]
    if reference.empty:
        raise ValueError(f"reference estimator {reference_estimator!r} not in results")

    reference_loss = reference.set_index(keys)[loss]
    merged = aggregated.join(
        reference_loss.rename("reference_loss"), on=keys, how="left"
    )
    merged["loss_excess"] = merged[loss] - merged["reference_loss"]
    merged["within_epsilon"] = merged["loss_excess"] <= epsilon
    merged["epsilon"] = epsilon
    return merged


# ---------------------------------------------------------------------------
# Storage and resumption
# ---------------------------------------------------------------------------


def write_raw(rows: list[dict[str, object]], path: Path | str) -> Path:
    """Append replicate rows to a Parquet file, creating it if absent."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)

    if target.exists():
        existing = pd.read_parquet(target)
        frame = pd.concat([existing, frame], ignore_index=True)

    # Object columns holding lists (beta vectors) survive Parquet only as
    # strings; keeping them readable matters more than round-tripping the type.
    for column in frame.columns:
        if frame[column].map(lambda v: isinstance(v, (list, tuple))).any():
            frame[column] = frame[column].map(
                lambda v: ",".join(map(str, v)) if isinstance(v, (list, tuple)) else v
            )

    frame.to_parquet(target, index=False)
    return target


def read_raw(path: Path | str) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    return pd.read_parquet(target)


def completed_cells(path: Path | str) -> set[tuple[str, str, int]]:
    """Which ``(scenario, estimator, replication)`` triples are already done.

    Resumption is by identity, not by count. Because seeds are derived from
    identifiers, a resumed run reproduces exactly the data an interrupted one
    would have produced, so a partially complete experiment is not tainted.
    """
    frame = read_raw(path)
    if frame.empty:
        return set()
    return {
        (str(row.scenario_id), str(row.estimator_id), int(row.replication_id))
        for row in frame.itertuples()
    }
