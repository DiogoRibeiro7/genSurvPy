from typing import Literal, Sequence, TypedDict, cast

import numpy as np
import pandas as pd

from gen_surv._rng import RandomStateLike, resolve_rng
from gen_surv._truth import current, record
from gen_surv.baseline import ExponentialBaseline
from gen_surv.censoring import CensoringFunc
from gen_surv.multistate import Transition, gen_multistate
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

    # Every intensity is constant in time, which is what "time-homogeneous"
    # means, so the baseline is exponential and the clock makes no difference:
    # a constant hazard is memoryless.
    transitions = [
        Transition(
            origin,
            destination,
            ExponentialBaseline(rate=float(intensity)),
            [float(coefficient)],
        )
        for (origin, destination), intensity, coefficient in (
            ((1, 2), rate[0], beta[0]),
            ((1, 3), rate[1], beta[1]),
            ((2, 3), rate[2], beta[2]),
        )
    ]

    data = gen_multistate(
        n=n,
        transitions=transitions,
        clock="forward",
        initial_state=1,
        covariate_dist="uniform",
        covariate_params={"low": 0.0, "high": float(covariate_range)},
        # Validated above against the same two choices the engine
        # accepts; the cast tells the type checker what that check
        # already guarantees.
        model_cens=cast(Literal["uniform", "exponential"], model_cens),
        cens_par=cens_par,
        layout="panel",
        seed=seed,
    )

    # This model has numbered its subjects from 1 since it was ported.
    data["id"] = data["id"] + 1
    _record_transition_times(beta, rate)
    return data[["id", "time", "state", "X0"]]


def _record_transition_times(beta: Sequence[float], rate: Sequence[float]) -> None:
    """Translate the engine's latent times into this model's vocabulary."""
    sink = current()
    if sink is None:
        return

    latent = sink.get("latent_times", {})
    if not latent:
        return

    entry_to_2 = np.asarray(latent[(1, 2)], dtype=float)
    record(
        beta=np.asarray(beta, dtype=float),
        rate=np.asarray(rate, dtype=float),
        transition_times={
            "t12": entry_to_2,
            "t13": np.asarray(latent[(1, 3)], dtype=float),
            "t23": np.asarray(latent[(2, 3)], dtype=float) - entry_to_2,
        },
    )
