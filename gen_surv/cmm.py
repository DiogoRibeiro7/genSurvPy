from typing import Sequence, TypedDict

import numpy as np
import pandas as pd

from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validation import validate_gen_cmm_inputs


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
    seed: int | None = None,
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
        DataFrame with columns: ``id``, ``start``, ``stop``, ``status``, ``X0``, ``transition``.

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
    >>> df.head()
    """
    validate_gen_cmm_inputs(n, model_cens, cens_par, beta, covariate_range, rate)

    rng = np.random.default_rng(seed)
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens

    z1 = rng.uniform(0, covariate_range, size=n)
    c = rfunc(n, cens_par, rng)

    u = rng.uniform(size=(3, n))
    t12 = (-np.log(1 - u[0]) / (rate[0] * np.exp(beta[0] * z1))) ** (1 / rate[1])
    t13 = (-np.log(1 - u[1]) / (rate[2] * np.exp(beta[1] * z1))) ** (1 / rate[3])

    first_event = np.minimum(t12, t13)
    censored = first_event >= c

    status = (~censored).astype(int)
    transition = np.where(censored, np.nan, np.where(t12 <= t13, 1, 2))
    stop = np.where(censored, c, first_event)

    return pd.DataFrame(
        {
            "id": np.arange(1, n + 1),
            "start": np.zeros(n),
            "stop": stop,
            "status": status,
            "X0": z1,
            "transition": transition,
        }
    )
