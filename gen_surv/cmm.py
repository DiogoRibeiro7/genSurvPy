from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from gen_surv._rng import RandomStateLike, resolve_rng
from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validation import validate_gen_cmm_inputs

_COLUMNS = ["id", "start", "stop", "from_state", "to_state", "status", "X0"]


class EventTimes(TypedDict):
    t12: float
    t13: float
    t23: float


def generate_event_times(
    z1: float,
    beta: Sequence[float],
    rate: Sequence[float],
    rng: np.random.Generator | None = None,
) -> EventTimes:
    """Generate event times for a continuous-time multi-state Markov model.

    Parameters
    ----------
    z1 : float
        Covariate value.
    beta : Sequence[float]
        List of 3 beta coefficients.
    rate : Sequence[float]
        List of 6 transition rate parameters.
    rng : np.random.Generator, optional
        Random number generator to use. Defaults to ``None`` which creates a new generator.

    Returns
    -------
    EventTimes
        Dictionary with keys ``'t12'``, ``'t13'``, and ``'t23'``.

    Examples
    --------
    >>> from gen_surv.cmm import generate_event_times
    >>> ev = generate_event_times(0.2, [0.1, -0.2, 0.3],
    ...                          [0.5, 1.0, 0.7, 1.2, 0.4, 1.5])
    >>> sorted(ev.keys())
    ['t12', 't13', 't23']
    """
    rng = np.random.default_rng() if rng is None else rng

    u = rng.uniform(size=3)
    rate_arr = np.asarray(rate).reshape(3, 2)
    beta_arr = np.asarray(beta)
    t = (-np.log(1 - u) / (rate_arr[:, 0] * np.exp(beta_arr * z1))) ** (
        1 / rate_arr[:, 1]
    )

    return {"t12": float(t[0]), "t13": float(t[1]), "t23": float(t[2])}


def gen_cmm(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
    seed: RandomStateLike = None,
) -> pd.DataFrame:
    """Generate survival data using a continuous-time Markov model (CMM).

    Parameters
    ----------
    n : int
        Number of individuals.
    model_cens : str
        ``"uniform"`` or ``"exponential"``.
    cens_par : float
        Parameter for censoring.
    beta : Sequence[float]
        Regression coefficients (length 3).
    covariate_range : float
        Upper bound for the covariate values.
    rate : Sequence[float]
        Transition rates (length 6).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Counting-process records with columns ``id``, ``start``, ``stop``,
        ``from_state``, ``to_state``, ``status``, ``X0``, sorted by ``id``,
        ``start`` then ``to_state``.

        States are 1 (healthy), 2 (illness) and 3 (death). While a subject
        occupies state 1 it is simultaneously at risk of ``1 -> 2`` and
        ``1 -> 3``, so it contributes one row for each, both ending when it
        leaves state 1; ``status`` is 1 on the transition that occurred and 0 on
        the competing one. A subject that reaches state 2 contributes a further
        ``2 -> 3`` row. Subjects therefore contribute two or three rows each,
        not one.

    Notes
    -----
    Sojourn times are drawn on a reset clock, so the model is semi-Markov: the
    ``2 -> 3`` row spans ``t12`` to ``t12 + t23`` where ``t23`` is an
    independent draw. This matches ``genCMM`` in the R package.

    Examples
    --------
    >>> from gen_surv.cmm import gen_cmm
    >>> df = gen_cmm(
    ...     n=50,
    ...     model_cens="uniform",
    ...     cens_par=2.0,
    ...     beta=[0.3, -0.2, 0.1],
    ...     covariate_range=1.0,
    ...     rate=[0.1, 1.0, 0.2, 1.2, 0.3, 1.5],
    ...     seed=42,
    ... )
    >>> list(df.columns)
    ['id', 'start', 'stop', 'from_state', 'to_state', 'status', 'X0']
    """
    validate_gen_cmm_inputs(n, model_cens, cens_par, beta, covariate_range, rate)

    rng = resolve_rng(seed)
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens

    z1 = rng.uniform(0, covariate_range, size=n)
    c = rfunc(n, cens_par, rng)

    # Latent transition times. All three rate pairs and all three coefficients
    # are used: releases up to 1.3.0 drew t23 and discarded it, leaving
    # rate[4], rate[5] and beta[2] with no effect on the output.
    u = rng.uniform(size=(3, n))
    t12 = (-np.log(1 - u[0]) / (rate[0] * np.exp(beta[0] * z1))) ** (1 / rate[1])
    t13 = (-np.log(1 - u[1]) / (rate[2] * np.exp(beta[1] * z1))) ** (1 / rate[3])
    t23 = (-np.log(1 - u[2]) / (rate[4] * np.exp(beta[2] * z1))) ** (1 / rate[5])

    # Ties go to the event, matching the R implementation's `c < min(t12, t13)`.
    censored_in_1 = c < np.minimum(t12, t13)
    illness_first = ~censored_in_1 & (t12 <= t13)
    death_first = ~censored_in_1 & (t13 < t12)

    exit_1 = np.where(censored_in_1, c, np.minimum(t12, t13))
    ids = np.arange(n)
    zeros = np.zeros(n)

    # Both competing transitions out of state 1 are observed until the subject
    # leaves it, so each contributes a row over the same interval.
    to_2 = pd.DataFrame(
        {
            "id": ids,
            "start": zeros,
            "stop": exit_1,
            "from_state": 1,
            "to_state": 2,
            "status": illness_first.astype(int),
            "X0": z1,
        }
    )
    to_3 = pd.DataFrame(
        {
            "id": ids,
            "start": zeros,
            "stop": exit_1,
            "from_state": 1,
            "to_state": 3,
            "status": death_first.astype(int),
            "X0": z1,
        }
    )

    # Only subjects that reached state 2 are at risk of 2 -> 3.
    reached_2 = np.flatnonzero(illness_first)
    entry = t12[reached_2]
    death_23 = entry + t23[reached_2]
    censored_in_2 = c[reached_2] < death_23
    from_2 = pd.DataFrame(
        {
            "id": ids[reached_2],
            "start": entry,
            "stop": np.where(censored_in_2, c[reached_2], death_23),
            "from_state": 2,
            "to_state": 3,
            "status": (~censored_in_2).astype(int),
            "X0": z1[reached_2],
        }
    )

    data = pd.concat([to_2, to_3, from_2], ignore_index=True)
    data = data.sort_values(["id", "start", "to_state"], kind="stable")
    return data.reset_index(drop=True)[_COLUMNS]
