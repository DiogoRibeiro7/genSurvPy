"""Drawing replicates from ``gen_surv``, with reproducible seeds and calibrated censoring.

Two problems this module solves that the generators themselves do not.

**Seeds must come from identifiers, not from execution order.** A study that
seeds replicate *k* from a counter produces different data when it is run in
parallel, or resumed after a crash, or when one scenario is added in the
middle. Here the seed is a pure function of
``(master_seed, scenario_id, replication_id)``, so any subset can be rerun in
any order and reproduce exactly.

**Censoring is specified as a rate, not as a parameter.** The design asks for
10%, 30%, 50% and 70% censoring, but the generators take ``cens_par`` -- the
upper bound of a uniform draw, or the mean of an exponential one -- whose
relationship to the realised rate depends on the DGP, the covariate effect and
the time scale. :func:`calibrate_censoring` inverts that numerically, once per
scenario, and the result is frozen into the configuration. Recalibrating per
replicate would make the mechanism itself random.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gen_surv import simulate as gen_surv_simulate

__all__ = [
    "derive_seed",
    "CensoringCalibration",
    "calibrate_censoring",
    "Replicate",
    "draw_replicate",
]

#: Enough draws that the calibrated rate is stable to well under a percentage
#: point, without making calibration a noticeable share of runtime.
CALIBRATION_N = 20000

#: The search bracket for ``cens_par``. Wide, because time scales differ by
#: orders of magnitude between DGPs; the bisection narrows it quickly.
CENS_PAR_LOW = 1e-4
CENS_PAR_HIGH = 1e6


def derive_seed(
    master_seed: int, scenario_id: str, replication_id: int, stream: str = "train"
) -> int:
    """A reproducible seed determined entirely by its identifiers.

    The scenario id is hashed rather than enumerated so that adding, removing
    or reordering scenarios does not change the seed of any other scenario.
    That property is what lets the design be extended without invalidating
    results already computed.

    ``stream`` separates the training draw from the independent evaluation
    draw within the same replication, so the two are never the same data and
    neither depends on the order they are generated in.
    """
    digest = hashlib.sha256(
        f"{scenario_id}|{replication_id}|{stream}".encode("utf-8")
    ).digest()[:8]
    sequence = np.random.SeedSequence(
        entropy=[int(master_seed), int.from_bytes(digest, "big")]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


@dataclass(frozen=True)
class CensoringCalibration:
    """The ``cens_par`` that realises a target censoring rate, and how well."""

    cens_par: float
    achieved: float
    target: float
    feasible: bool
    reason: str = ""

    @property
    def error(self) -> float:
        return abs(self.achieved - self.target)


def _censoring_rate(
    dgp: str, params: Mapping[str, Any], cens_par: float, n: int, seed: int
) -> float:
    frame = gen_surv_simulate(
        dgp, n=n, **{**params, "cens_par": float(cens_par)}, seed=seed
    ).data
    return float((frame["status"] == 0).mean())


def calibrate_censoring(
    dgp: str,
    params: Mapping[str, Any],
    target: float,
    *,
    n: int = CALIBRATION_N,
    seed: int = 987654321,
    tolerance: float = 0.005,
    max_iterations: int = 60,
) -> CensoringCalibration:
    """Find ``cens_par`` giving a censoring rate of ``target``, by bisection.

    The rate is monotonically decreasing in ``cens_par`` for both censoring
    families -- a later censoring time can only turn a censored observation
    into an observed one -- which is what makes bisection valid. The function
    is a step function of a finite sample, so the search is on a fixed
    calibration seed and stops at ``tolerance``.

    Infeasibility is reported, not worked around. ``mixture_cure`` is the case
    that matters: a cured subject never fails, so the censoring rate cannot go
    below the cure fraction however late censoring is applied. A scenario
    asking for 10% censoring under a 30% cure fraction is not a scenario, and
    the pilot should drop it rather than silently return the closest thing.
    """
    if not 0.0 <= target < 1.0:
        raise ValueError("target censoring must be in [0, 1)")

    low, high = CENS_PAR_LOW, CENS_PAR_HIGH
    rate_at_high = _censoring_rate(dgp, params, high, n, seed)
    if rate_at_high > target + tolerance:
        return CensoringCalibration(
            cens_par=high,
            achieved=rate_at_high,
            target=target,
            feasible=False,
            reason=(
                f"censoring cannot fall below {rate_at_high:.3f} for this DGP even "
                f"with cens_par={high:g}; the mechanism itself leaves subjects "
                f"without an event (a cure fraction, for instance)"
            ),
        )

    rate_at_low = _censoring_rate(dgp, params, low, n, seed)
    if rate_at_low < target - tolerance:
        return CensoringCalibration(
            cens_par=low,
            achieved=rate_at_low,
            target=target,
            feasible=False,
            reason=(
                f"censoring cannot exceed {rate_at_low:.3f} for this DGP even with "
                f"cens_par={low:g}"
            ),
        )

    achieved = rate_at_high
    midpoint = high
    for _ in range(max_iterations):
        midpoint = float(np.sqrt(low * high))  # geometric: the scale spans decades
        achieved = _censoring_rate(dgp, params, midpoint, n, seed)
        if abs(achieved - target) <= tolerance:
            break
        if achieved > target:
            low = midpoint
        else:
            high = midpoint

    return CensoringCalibration(
        cens_par=midpoint,
        achieved=achieved,
        target=target,
        feasible=abs(achieved - target) <= tolerance,
        reason=(
            "" if abs(achieved - target) <= tolerance else "bisection did not converge"
        ),
    )


@dataclass(frozen=True)
class Replicate:
    """One simulated dataset with the truth that generated it."""

    scenario_id: str
    replication_id: int
    seed: int
    data: pd.DataFrame
    truth: Mapping[str, Any]
    params: Mapping[str, Any]
    dgp: str

    @property
    def covariates(self) -> NDArray[np.float64]:
        """The design matrix, as a 2-D array even when there is one covariate."""
        matrix = np.asarray(self.truth["covariates"], dtype=float)
        return matrix.reshape(len(self.data), -1)

    @property
    def observed_time(self) -> NDArray[np.float64]:
        return self.data["time"].to_numpy(dtype=float)

    @property
    def event(self) -> NDArray[np.bool_]:
        return self.data["status"].to_numpy() == 1

    @property
    def censoring_rate(self) -> float:
        return float(1.0 - self.event.mean())


def draw_replicate(
    dgp: str,
    params: Mapping[str, Any],
    n: int,
    scenario_id: str,
    replication_id: int,
    master_seed: int,
    stream: str = "train",
) -> Replicate:
    """Draw one replicate, seeded from its identifiers alone.

    ``stream="eval"`` draws the independent evaluation sample used to measure
    performance. Fitting and evaluating on the same data would confound
    misspecification -- the subject of this study -- with overfitting, which is
    not: a random survival forest scores a far higher apparent concordance than
    Cox on a correctly specified Cox mechanism purely by fitting noise.
    """
    seed = derive_seed(master_seed, scenario_id, replication_id, stream)
    result = gen_surv_simulate(dgp, n=n, **params, seed=seed)

    return Replicate(
        scenario_id=scenario_id,
        replication_id=replication_id,
        seed=seed,
        data=result.data,
        truth=result.truth,
        params=dict(params),
        dgp=dgp,
    )
