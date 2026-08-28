from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from gen_surv._rng import RandomStateLike, resolve_rng
from gen_surv._truth import record
from gen_surv.bivariate import sample_bivariate_distribution
from gen_surv.censoring import CensoringFunc, rexpocens, runifcens
from gen_surv.validation import ParameterError, validate_gen_tdcm_inputs


def generate_censored_observations(
    n: int,
    dist_par: Sequence[float],
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    lam: float,
    b: NDArray[np.float64],
    seed: RandomStateLike = None,
) -> NDArray[np.float64]:
    """Generate censored TDCM observations.

    Parameters
    ----------
    n : int
        Number of individuals.
    dist_par : Sequence[float]
        Not directly used here (kept for API compatibility).
    model_cens : {"uniform", "exponential"}
        Censoring model.
    cens_par : float
        Parameter for the censoring model.
    beta : Sequence[float]
        Length-2 list of regression coefficients.
    lam : float
        Rate parameter.
    b : NDArray[np.float64]
        Covariate matrix with two columns ``[., z1]``.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(n, 6)`` with columns
        ``[id, start, stop, status, covariate1 (z1), covariate2 (z2)]``.
    """
    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens
    rng = resolve_rng(seed)

    z1 = b[:, 1]

    # A heavy-tailed covariate distribution -- a Weibull `dist_par` shape well
    # below 1 -- puts `z1` in the tens of thousands, and `exp(beta[0] * z1)`
    # then leaves the range of a float. Neither outcome is survivable:
    # overflow to `inf` makes `t1 = log_term / inf` exactly 0.0, which
    # `status = (t <= c)` reports as an *observed event at time zero* in a
    # zero-length risk interval; underflow to 0.0 makes `t` infinite, silently
    # reported as censored. Both are frames of the right shape carrying data no
    # analysis should be handed, so this raises rather than returning one.
    # Errors are suppressed only because the result is inspected immediately
    # below and turned into a message that says what to change; NumPy's warning
    # names the expression, not the parameter behind it.
    with np.errstate(over="ignore", under="ignore"):
        exp_b0_z1 = np.exp(beta[0] * z1)
    if not np.all(np.isfinite(exp_b0_z1)) or np.any(exp_b0_z1 == 0.0):
        extreme = float(z1[np.argmax(np.abs(beta[0] * z1))])
        raise ParameterError(
            "beta",
            beta,
            f"combined with the covariate distribution, exp(beta[0] * z) left "
            f"the range of a float (largest covariate drawn: {extreme:.3g}). "
            f"Reduce beta[0], or widen the Weibull shape in dist_par -- a "
            f"shape below 1 is heavy-tailed and draws very large covariates",
        )

    x = lam * b[:, 0] * exp_b0_z1
    u = rng.uniform(size=n)
    c = rfunc(n, cens_par, rng)

    threshold = 1 - np.exp(-x)
    log_term = -np.log(1 - u)

    # Before the crossover the hazard is lam * exp(beta[0] * z1); after it, that
    # times exp(beta[1]). Inverting the cumulative hazard on each side:
    #
    #   before:  t = L / A
    #   after:   t = tau + (L - x) / (A * exp(beta[1]))
    #
    # with A = lam * exp(beta[0] * z1), tau = x / A the crossover time, x the
    # cumulative hazard accrued by then, and L = -log(1 - u). Expanding the
    # second gives the closed form below. Releases up to 2.0.2 had the sign of
    # the x term reversed, which placed "after the crossover" draws *before* it
    # and, for large beta[1], produced negative survival times.
    t1 = log_term / (lam * exp_b0_z1)
    t2 = (log_term + x * (np.exp(beta[1]) - 1)) / (lam * np.exp(beta[0] * z1 + beta[1]))
    mask = u < threshold
    t = np.where(mask, t1, t2)

    # The covariate's value over the interval actually observed: a subject
    # censored before its crossover never switched, whatever its latent event
    # time would have done.
    crossover = b[:, 0]
    z2 = (crossover <= np.minimum(t, c)).astype(float)

    time = np.minimum(t, c)
    status = (t <= c).astype(float)

    # The crossover time is what the returned frame cannot express: it records
    # only the covariate's value at exit, so a caller cannot split the risk
    # interval without this.
    record(
        beta=np.asarray(beta, dtype=float),
        covariates=z1,
        crossover_time=b[:, 0],
        event_time=t,
        censoring_time=c,
        switched_before_exit=z2,
    )

    ids = np.arange(1, n + 1, dtype=float)
    zeros = np.zeros(n, dtype=float)
    return np.column_stack((ids, zeros, time, status, z1, z2))


def gen_tdcm(
    n: int,
    dist: str,
    corr: float,
    dist_par: Sequence[float],
    model_cens: str,
    cens_par: float,
    beta: Sequence[float],
    lam: float,
    seed: RandomStateLike = None,
) -> pd.DataFrame:
    """Generate TDCM (Time-Dependent Covariate Model) survival data.

    Parameters
    ----------
    n : int
        Number of individuals.
    dist : {"weibull", "exponential"}
        Type of marginal distributions.
    corr : float
        Correlation between the baseline covariate and the crossover time, on
        the latent normal scale. Must be in ``(0, 1)`` for ``dist='weibull'``
        and ``(-1, 1)`` for ``dist='exponential'``; the endpoints make the
        copula's covariance singular.
    dist_par : Sequence[float]
        Distribution parameters.
    model_cens : {"uniform", "exponential"}
        Censoring model.
    cens_par : float
        Censoring parameter.
    beta : Sequence[float]
        Length-2 regression coefficients: the baseline covariate effect and the
        effect of the time-dependent covariate.
    lam : float
        Lambda rate parameter.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns are ``["id", "start", "stop", "status", "covariate", "tdcov"]``.

    Examples
    --------
    >>> from gen_surv.tdcm import gen_tdcm
    >>> df = gen_tdcm(
    ...     n=5,
    ...     dist="exponential",
    ...     corr=0.3,
    ...     dist_par=[0.5, 1.0],
    ...     model_cens="uniform",
    ...     cens_par=2.0,
    ...     beta=[0.1, 0.2],
    ...     lam=0.5,
    ...     seed=42,
    ... )
    """
    validate_gen_tdcm_inputs(n, dist, corr, dist_par, model_cens, cens_par, beta, lam)

    # One generator shared by both stages, so a single seed reproduces the
    # covariates and the event/censoring times together.
    rng = resolve_rng(seed)

    # Generate covariate matrix from bivariate distribution
    b = sample_bivariate_distribution(n, dist, corr, dist_par, rng)

    data = generate_censored_observations(
        n, dist_par, model_cens, cens_par, beta, lam, b, rng
    )

    return pd.DataFrame(
        data, columns=["id", "start", "stop", "status", "covariate", "tdcov"]
    )
