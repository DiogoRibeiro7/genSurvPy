from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from gen_surv._rng import RandomStateLike, resolve_rng
from gen_surv._truth import record
from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validation import validate_gen_thmm_inputs


class TransitionTimes(TypedDict):
    c: float
    t12: float
    t13: float
    t23: float


def calculate_transitions(
    z1: float,
    cens_par: float,
    beta: Sequence[float],
    rate: Sequence[float],
    rfunc: CensoringFunc,
    seed: RandomStateLike = None,
) -> TransitionTimes:
    """
    Calculate transition and censoring times for THMM.

    Parameters:
    - z1 (float): Covariate value.
    - cens_par (float): Censoring parameter.
    - beta (list of float): Coefficients for rate modification (length 3).
    - rate (list of float): Base rates (length 3).
    - rfunc (callable): Censoring function, e.g. runifcens or rexpocens.
    - seed (int, Generator or None): Seed or generator for reproducibility.

    Returns:
    - dict with keys 'c', 't12', 't13', 't23'
    """
    rng = resolve_rng(seed)

    c = rfunc(1, cens_par, rng)[0]
    rate12 = rate[0] * np.exp(beta[0] * z1)
    rate13 = rate[1] * np.exp(beta[1] * z1)
    rate23 = rate[2] * np.exp(beta[2] * z1)

    t12 = rng.exponential(scale=1 / rate12)
    t13 = rng.exponential(scale=1 / rate13)
    t23 = rng.exponential(scale=1 / rate23)

    return {"c": c, "t12": t12, "t13": t13, "t23": t23}


def gen_thmm(
    n: int,
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    covariate_range: float,
    rate: Sequence[float],
    seed: RandomStateLike = None,
) -> pd.DataFrame:
    """Generate THMM (Time-Homogeneous Markov Model) survival data.

    Parameters
    ----------
    n : int
        Number of individuals.
    model_cens : {"uniform", "exponential"}
        Censoring model.
    cens_par : float
        Censoring parameter.
    beta : Sequence[float]
        Length-3 regression coefficients.
    covariate_range : float
        Upper bound for the covariate values.
    rate : Sequence[float]
        Length-3 transition rates.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns = ``["id", "time", "state", "X0"]``, one row per observation
        time giving the state occupied at that time.

        States are 1 (healthy), 2 (illness) and 3 (death). Every subject starts
        with an observation in state 1 at time 0, then contributes one or two
        further observations, so subjects yield two or three rows each rather
        than one. A subject still in state 1 or 2 when censoring occurs has a
        final observation in that state at the censoring time.

    Notes
    -----
    This panel layout -- a state recorded at each observation time -- matches
    ``genTHMM`` in the R package, and differs deliberately from
    :func:`gen_surv.cmm.gen_cmm`, which emits counting-process intervals.

    All transition intensities are constant in time, so sojourn times are
    exponential and the reset and forward clocks coincide.

    Examples
    --------
    >>> from gen_surv.thmm import gen_thmm
    >>> df = gen_thmm(
    ...     n=3,
    ...     model_cens="uniform",
    ...     cens_par=5.0,
    ...     beta=[0.1, 0.2, 0.3],
    ...     covariate_range=1.0,
    ...     rate=[0.1, 0.1, 0.2],
    ...     seed=42,
    ... )
    """
    validate_gen_thmm_inputs(n, model_cens, cens_par, beta, covariate_range, rate)
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens
    rng = resolve_rng(seed)
    records = []
    # Collected for the ground-truth report; they do not affect any draw.
    latent = {key: np.empty(n, dtype=float) for key in ("t12", "t13", "t23", "c")}
    covariates = np.empty(n, dtype=float)

    for k in range(n):
        z1 = rng.uniform(0, covariate_range)
        trans = calculate_transitions(z1, cens_par, beta, rate, rfunc, rng)
        t12, t13, t23, c = trans["t12"], trans["t13"], trans["t23"], trans["c"]

        covariates[k] = z1
        for key, value in (("t12", t12), ("t13", t13), ("t23", t23), ("c", c)):
            latent[key][k] = value

        # Every trajectory is observed in state 1 at entry.
        records.append([k + 1, 0.0, 1, z1])

        # Ties go to the event, matching the R implementation.
        if c < min(t12, t13):
            # Still healthy when censoring occurs.
            records.append([k + 1, c, 1, z1])
        elif t13 < t12:
            # Died without passing through the illness state.
            records.append([k + 1, t13, 3, z1])
        else:
            # Fell ill, then either died or was censored while ill. Releases up
            # to 1.3.0 stopped here and discarded t23, so the 2 -> 3 transition
            # never appeared and rate[2]/beta[2] had no effect on the output.
            records.append([k + 1, t12, 2, z1])
            if c < t12 + t23:
                records.append([k + 1, c, 2, z1])
            else:
                records.append([k + 1, t12 + t23, 3, z1])

    record(
        beta=np.asarray(beta, dtype=float),
        rate=np.asarray(rate, dtype=float),
        covariates=covariates,
        censoring_time=latent["c"],
        transition_times={key: latent[key] for key in ("t12", "t13", "t23")},
    )

    return pd.DataFrame(records, columns=["id", "time", "state", "X0"])
