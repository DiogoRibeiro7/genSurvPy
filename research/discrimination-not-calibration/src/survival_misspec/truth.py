"""True conditional survival functions for each data-generating mechanism.

This module is the reason the study is possible. Ordinary survival benchmarks
compare predictions against censored observed outcomes, which conflates the
error of the model with the noise of a single realisation. Here the mechanism
is known, so the target

.. math::

    S_i(t) = P(T_i > t \\mid X_i)

can be written down exactly and predictions compared against it directly.

Each generator needs its own expression -- there is no single formula that
covers them all, and assuming otherwise is the easiest way to produce a study
that measures its own algebra mistake. Every function below states the
derivation from the generator's own sampling code, and
``tests/test_truth.py`` checks each one against simulated draws by the
probability integral transform: if :math:`S_i` is right then
:math:`S_i(T_i) \\sim \\mathrm{Uniform}(0, 1)`, which tests the whole
distribution rather than a moment.

Conventions
-----------
Every ``survival`` function takes the recorded ``truth`` mapping from
``gen_surv.simulate`` and the DGP's own parameters, and returns an array of
shape ``(n_subjects, n_times)``. Time zero maps to 1.0 and the functions are
non-increasing in ``t`` by construction.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "TRUTH_FUNCTIONS",
    "SUPPORTED_DGPS",
    "true_survival",
    "unsupported_reason",
]

Truth = Mapping[str, Any]
Params = Mapping[str, Any]


def _as_times(times: NDArray[np.float64] | list[float]) -> NDArray[np.float64]:
    grid = np.asarray(times, dtype=float).reshape(1, -1)
    if np.any(grid < 0):
        raise ValueError("evaluation times must be non-negative")
    return grid


def _linear_predictor(truth: Truth) -> NDArray[np.float64]:
    return np.asarray(truth["linear_predictor"], dtype=float).reshape(-1, 1)


# ---------------------------------------------------------------------------
# cphm
# ---------------------------------------------------------------------------


def _survival_cphm(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """Exponential baseline with a proportional covariate effect.

    ``gen_cphm`` draws ``x = rng.exponential(scale=1 / exp(beta * z))``, so the
    hazard is constant at :math:`e^{\\beta z}` and

    .. math::

        S(t \\mid z) = \\exp\\!\\left(-t\\, e^{\\beta z}\\right).

    The covariate is uniform on ``[0, covariate_range]``, which matters for the
    marginal distribution but not for this conditional one.
    """
    grid = _as_times(times)
    hazard = np.exp(_linear_predictor(truth))
    return np.exp(-grid * hazard)


# ---------------------------------------------------------------------------
# aft_weibull
# ---------------------------------------------------------------------------


def _survival_aft_weibull(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """Weibull, which is the one family that is both proportional-hazards and AFT.

    ``gen_aft_weibull`` draws ``T = scale * (-log(U) * exp(-eta)) ** (1 / shape)``
    with ``U`` uniform, so with :math:`E = -\\log U \\sim \\mathrm{Exp}(1)`

    .. math::

        P(T > t)
        = P\\!\\left(E > e^{\\eta} (t / \\text{scale})^{\\text{shape}}\\right)
        = \\exp\\!\\left(-e^{\\eta} (t/\\text{scale})^{\\text{shape}}\\right).

    The linear predictor multiplies the cumulative hazard, so a Cox model is
    correctly specified for the *ranking* here even though the baseline is
    parametric.
    """
    grid = _as_times(times)
    shape = float(params["shape"])
    scale = float(params["scale"])
    eta = _linear_predictor(truth)
    return np.exp(-np.exp(eta) * (grid / scale) ** shape)


# ---------------------------------------------------------------------------
# aft_ln
# ---------------------------------------------------------------------------


def _survival_aft_log_normal(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """Log-normal AFT, which is **not** proportional hazards.

    ``gen_aft_log_normal`` sets ``log_T = X @ beta + epsilon`` with
    ``epsilon ~ Normal(0, sigma)``, so

    .. math::

        S(t \\mid X)
        = P(\\log T > \\log t)
        = \\Phi\\!\\left(\\frac{X^\\top\\beta - \\log t}{\\sigma}\\right).

    The hazard rises then falls, and hazard ratios between two covariate values
    change with ``t``. This is the primary misspecified case for Cox.
    """
    grid = _as_times(times)
    sigma = float(params["sigma"])
    eta = _linear_predictor(truth)

    with np.errstate(divide="ignore"):
        log_t = np.log(grid)

    survival = stats.norm.cdf((eta - log_t) / sigma)
    # log(0) = -inf above gives cdf(inf) = 1, which is the correct S(0) = 1.
    return np.asarray(survival, dtype=float)


# ---------------------------------------------------------------------------
# aft_log_logistic
# ---------------------------------------------------------------------------


def _survival_aft_log_logistic(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """Log-logistic, with a unimodal hazard and non-proportional hazards.

    ``gen_aft_log_logistic`` draws
    ``T = scale * (U / (1 - U)) ** (1 / shape) * exp(-eta / shape)``. Writing
    :math:`A = \\text{scale}\\, e^{-\\eta/\\text{shape}}` and
    :math:`v = (t/A)^{\\text{shape}}`, the event :math:`T > t` is
    :math:`U/(1-U) > v`, that is :math:`U > v/(1+v)`, so

    .. math::

        S(t \\mid X) = \\frac{1}{1 + v}
        = \\frac{1}{1 + e^{\\eta}\\,(t/\\text{scale})^{\\text{shape}}}.

    The generator clips ``U`` to ``[0.001, 0.999]`` before transforming it.
    This is a winsorised event-time distribution with endpoint atoms, not the
    untruncated log-logistic law. With
    ``q(t) = v / (1 + v)``, the implemented survival is 1 below the lower
    generated endpoint, ``1 - q(t)`` on the interior, and 0 at and above the
    upper generated endpoint.
    """
    grid = _as_times(times)
    shape = float(params["shape"])
    scale = float(params["scale"])
    eta = _linear_predictor(truth)
    odds = np.exp(eta) * (grid / scale) ** shape
    quantile = odds / (1.0 + odds)
    lower = 0.001
    upper = 0.999
    tolerance = 1e-12
    survival = 1.0 - quantile
    survival = np.where(quantile < lower - tolerance, 1.0, survival)
    survival = np.where(quantile >= upper - tolerance, 0.0, survival)
    return np.asarray(survival, dtype=float)


# ---------------------------------------------------------------------------
# piecewise_exponential
# ---------------------------------------------------------------------------


def _survival_piecewise_exponential(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """Constant hazard within intervals, scaled by the covariate effect.

    The generator adjusts every interval's rate by :math:`e^{\\eta}` and
    inverts the cumulative hazard, so with breakpoints
    :math:`0 = b_0 < b_1 < \\dots < b_K = \\infty` and rates :math:`h_k`,

    .. math::

        H(t \\mid X) = e^{\\eta} \\sum_k h_k \\,
        \\bigl|[0, t] \\cap [b_{k-1}, b_k)\\bigr|,
        \\qquad S = e^{-H}.

    Proportional hazards holds, with a baseline no parametric model in this
    study can represent exactly -- which is the point of including it.
    """
    grid = _as_times(times)
    breakpoints = np.asarray(truth["breakpoints"], dtype=float)
    rates = np.asarray(truth["hazard_rates"], dtype=float)
    eta = _linear_predictor(truth)

    edges = np.concatenate([[0.0], breakpoints, [np.inf]])
    widths = np.clip(grid[..., None] - edges[:-1], 0.0, None)
    widths = np.minimum(widths, np.diff(edges))

    baseline_cumulative = (widths * rates).sum(axis=-1)
    return np.exp(-np.exp(eta) * baseline_cumulative)


# ---------------------------------------------------------------------------
# mixture_cure
# ---------------------------------------------------------------------------


def _survival_mixture_cure(
    times: NDArray[np.float64], truth: Truth, params: Params
) -> NDArray[np.float64]:
    """A cured subpopulation plus an exponential failure time.

    The generator draws cure status with
    :math:`\\pi(X) = \\mathrm{expit}(\\mathrm{logit}(\\text{cure\\_fraction}) +
    X^\\top\\beta_{\\text{cure}})` and, for the uncured, an exponential with
    rate :math:`h_0 e^{X^\\top\\beta_{\\text{surv}}}`. Marginalising over cure
    status,

    .. math::

        S(t \\mid X) = \\pi(X)
        + \\bigl(1 - \\pi(X)\\bigr)
        \\exp\\!\\left(-t\\, h_0 e^{X^\\top\\beta_{\\text{surv}}}\\right).

    This is the conditional law given covariates alone, which is what a model
    with access only to ``X`` can hope to predict -- not the law given the
    latent cure indicator. :math:`S` has a plateau at :math:`\\pi(X)` rather
    than decaying to zero, which no proportional-hazards model can reproduce.
    """
    grid = _as_times(times)
    baseline = float(params["baseline_hazard"])
    cure_fraction = float(params["cure_fraction"])

    cure_lp = np.asarray(truth["cure_linear_predictor"], dtype=float).reshape(-1, 1)
    logit = np.log(cure_fraction / (1.0 - cure_fraction))
    cure_probability = 1.0 / (1.0 + np.exp(-(logit + cure_lp)))

    hazard = baseline * np.exp(_linear_predictor(truth))
    uncured = np.exp(-grid * hazard)

    return cure_probability + (1.0 - cure_probability) * uncured


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TruthFunction = Callable[[NDArray[np.float64], Truth, Params], NDArray[np.float64]]

TRUTH_FUNCTIONS: dict[str, TruthFunction] = {
    "cphm": _survival_cphm,
    "aft_weibull": _survival_aft_weibull,
    "aft_ln": _survival_aft_log_normal,
    "aft_log_logistic": _survival_aft_log_logistic,
    "piecewise_exponential": _survival_piecewise_exponential,
    "mixture_cure": _survival_mixture_cure,
}

SUPPORTED_DGPS = tuple(sorted(TRUTH_FUNCTIONS))

#: Generators deliberately excluded from this study, with the reason. Kept as
#: data so the protocol and the paper can quote it rather than restate it.
EXCLUDED_DGPS: dict[str, str] = {
    "tdcm": (
        "The covariate switches value during follow-up, so S(t | X) conditional "
        "on baseline covariates alone requires marginalising over a latent "
        "crossover time. That is a coherent estimand but a different one from "
        "the single-event, baseline-covariate prediction task this study "
        "defines, and the estimators here are not given the time-varying "
        "structure. Reserved for later work."
    ),
    "competing_risks": (
        "Cause-specific survival is not a single-event estimand; the "
        "discrimination and calibration measures used here are defined for one "
        "event type. Out of scope by design."
    ),
    "competing_risks_weibull": "See competing_risks.",
    "recurrent_events": (
        "Multiple events per subject; the prediction target is an intensity, "
        "not a single survival function. Out of scope by design."
    ),
    "cmm": "Multi-state process; out of scope for a single-event study.",
    "thmm": "Multi-state process; out of scope for a single-event study.",
}


def unsupported_reason(dgp: str) -> str | None:
    """Why ``dgp`` is not part of this study, or ``None`` if it is."""
    if dgp in TRUTH_FUNCTIONS:
        return None
    return EXCLUDED_DGPS.get(dgp, f"No truth function is implemented for {dgp!r}.")


def true_survival(
    dgp: str,
    times: NDArray[np.float64] | list[float],
    truth: Truth,
    params: Params,
) -> NDArray[np.float64]:
    """Return ``S_i(t)`` with shape ``(n_subjects, n_times)``.

    Parameters
    ----------
    dgp:
        A ``gen_surv`` model name that this study supports.
    times:
        Evaluation grid, non-negative and typically increasing.
    truth:
        The ``truth`` mapping from :func:`gen_surv.simulate`.
    params:
        The DGP's own parameters, as passed to the generator. Needed because
        not every parameter is recorded in ``truth``.

    Raises
    ------
    KeyError
        If ``dgp`` has no truth function, with the recorded reason when the
        exclusion is deliberate.
    """
    function = TRUTH_FUNCTIONS.get(dgp)
    if function is None:
        raise KeyError(
            f"{dgp!r} is not supported by this study: {unsupported_reason(dgp)}"
        )

    grid = np.asarray(times, dtype=float)
    surface = function(grid, truth, params)

    # These hold by construction for every expression above; asserting them
    # here turns an algebra slip into an immediate failure rather than a
    # quietly wrong loss.
    if not np.all(np.isfinite(surface)):
        raise FloatingPointError(f"{dgp} true survival produced non-finite values")
    if np.any(surface < -1e-12) or np.any(surface > 1.0 + 1e-12):
        raise FloatingPointError(f"{dgp} true survival left [0, 1]")

    return np.clip(surface, 0.0, 1.0)
