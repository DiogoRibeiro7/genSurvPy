from typing import Literal, Sequence, TypedDict, cast

import numpy as np
import pandas as pd

from gen_surv._rng import RandomStateLike
from gen_surv._truth import current, record
from gen_surv.baseline import WeibullBaseline
from gen_surv.multistate import Transition, gen_multistate
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

    # `rate` is three (intensity, shape) pairs. The sojourn is drawn from
    # H(t) = lambda * t**rho, which is a Weibull cumulative hazard
    # (t / scale) ** shape with shape = rho and scale = lambda ** (-1 / rho).
    transitions = [
        Transition(
            origin,
            destination,
            WeibullBaseline(shape=shape, scale=float(intensity) ** (-1.0 / shape)),
            [float(coefficient)],
        )
        for (origin, destination), intensity, shape, coefficient in (
            ((1, 2), rate[0], float(rate[1]), beta[0]),
            ((1, 3), rate[2], float(rate[3]), beta[1]),
            ((2, 3), rate[4], float(rate[5]), beta[2]),
        )
    ]

    # A reset clock: the 2 -> 3 sojourn is measured from entry to state 2, which
    # is what makes this semi-Markov and what `genCMM` does in the R package.
    data = gen_multistate(
        n=n,
        transitions=transitions,
        clock="reset",
        initial_state=1,
        covariate_dist="uniform",
        covariate_params={"low": 0.0, "high": float(covariate_range)},
        # Validated above against the same two choices the engine
        # accepts; the cast tells the type checker what that check
        # already guarantees.
        model_cens=cast(Literal["uniform", "exponential"], model_cens),
        cens_par=cens_par,
        layout="intervals",
        seed=seed,
    )

    _record_transition_times(beta, rate)
    return data[_COLUMNS]


def _record_transition_times(beta: Sequence[float], rate: Sequence[float]) -> None:
    """Translate the engine's latent times into this model's vocabulary.

    The engine records the first candidate drawn for every edge, keyed by
    ``(origin, destination)``. Callers of :func:`gen_surv.simulate` know this
    model by ``t12``, ``t13`` and ``t23``, the last of which is a sojourn rather
    than an absolute time.
    """
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
            # Absolute in the engine, a sojourn here.
            "t23": np.asarray(latent[(2, 3)], dtype=float) - entry_to_2,
        },
    )
