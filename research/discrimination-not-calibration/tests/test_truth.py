"""The true survival functions, which everything else in the study depends on.

If ``S_i(t)`` is wrong then every loss in the paper measures an algebra mistake
rather than an estimator, and nothing downstream would reveal it: the numbers
would be finite, ordered plausibly, and completely meaningless.

The central test is the probability integral transform. If :math:`S_i` is the
true conditional survival function and :math:`T_i` is drawn from it, then
:math:`S_i(T_i) \\sim \\mathrm{Uniform}(0,1)` -- a statement about the whole
distribution, including the covariate effect, not about a moment. Comparing
means would pass for a function with the right centre and the wrong shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from survival_misspec.truth import (
    EXCLUDED_DGPS,
    SUPPORTED_DGPS,
    true_survival,
    unsupported_reason,
)

from gen_surv import simulate

#: Parameters per DGP, with censoring set so far out that essentially every
#: latent event time is observed. The transform is about the event-time law, so
#: censoring is a nuisance here even though it is a design factor elsewhere.
CASES: dict[str, dict] = {
    "cphm": {
        "beta": 0.6,
        "covariate_range": 2.0,
        "model_cens": "uniform",
        "cens_par": 50.0,
    },
    "aft_weibull": {
        "beta": [0.5, -0.3],
        "shape": 1.4,
        "scale": 2.0,
        "model_cens": "uniform",
        "cens_par": 500.0,
    },
    "aft_ln": {
        "beta": [0.5, -0.3],
        "sigma": 0.8,
        "model_cens": "uniform",
        "cens_par": 500.0,
    },
    "aft_log_logistic": {
        "beta": [0.5, -0.3],
        "shape": 1.6,
        "scale": 2.0,
        "model_cens": "uniform",
        "cens_par": 500.0,
    },
    "piecewise_exponential": {
        "breakpoints": [0.5, 1.5],
        "hazard_rates": [0.4, 0.9, 1.6],
        "betas": [0.5, -0.3],
    },
    "mixture_cure": {
        "cure_fraction": 0.3,
        "baseline_hazard": 0.7,
        "betas_survival": [0.5, -0.3],
        "betas_cure": [0.4, 0.2],
    },
}


def test_every_supported_dgp_has_a_case() -> None:
    """A truth function added without a test would go unvalidated."""
    assert set(CASES) == set(SUPPORTED_DGPS)


def test_every_excluded_dgp_states_a_reason() -> None:
    """Exclusions are decisions, and a decision without a reason is an omission."""
    for dgp, reason in EXCLUDED_DGPS.items():
        assert reason.strip(), f"{dgp} is excluded with no reason recorded"
        assert unsupported_reason(dgp) == reason


def _subset(truth: dict, index: np.ndarray, n: int) -> dict:
    """Take the same subjects out of every per-subject array in ``truth``."""
    return {
        key: (
            value[index]
            if isinstance(value, np.ndarray) and value.shape[:1] == (n,)
            else value
        )
        for key, value in truth.items()
    }


@pytest.mark.slow
@pytest.mark.parametrize("dgp", sorted(CASES))
def test_true_survival_passes_the_probability_integral_transform(dgp: str) -> None:
    """S_i(T_i) must be Uniform(0, 1) if S_i is right."""
    n = 20000
    params = CASES[dgp]
    result = simulate(dgp, n=n, **params, seed=20260828)

    event_time = np.asarray(result.truth["event_time"], dtype=float)
    keep = np.isfinite(event_time)
    if dgp == "mixture_cure":
        # A cured subject has T = infinity, so S_i(T_i) sits on the atom at
        # pi(X) rather than being uniform. Restrict to the uncured, where the
        # conditional law is continuous, and undo the mixture below.
        keep &= np.asarray(result.truth["cured"]) == 0

    index = np.flatnonzero(keep)
    times = event_time[index]

    values = np.empty(index.size, dtype=float)
    chunk = 2000
    for start in range(0, index.size, chunk):
        block = index[start : start + chunk]
        surface = true_survival(
            dgp, times[start : start + chunk], _subset(result.truth, block, n), params
        )
        values[start : start + chunk] = np.diag(surface)

    if dgp == "mixture_cure":
        cure_lp = np.asarray(result.truth["cure_linear_predictor"], dtype=float)[index]
        logit = np.log(params["cure_fraction"] / (1 - params["cure_fraction"]))
        cure_probability = 1.0 / (1.0 + np.exp(-(logit + cure_lp)))
        values = (values - cure_probability) / (1.0 - cure_probability)

    outcome = stats.kstest(values, "uniform")
    assert outcome.pvalue > 0.001, (
        f"{dgp}: S_i(T_i) is not Uniform(0,1) -- KS D={outcome.statistic:.5f}, "
        f"p={outcome.pvalue:.3g}. The truth function does not match the "
        f"generator, so every loss computed with it is meaningless."
    )


@pytest.mark.parametrize("dgp", sorted(CASES))
def test_true_survival_is_a_survival_function(dgp: str) -> None:
    """Starts at one, never increases, stays in [0, 1]."""
    params = CASES[dgp]
    result = simulate(dgp, n=200, **params, seed=7)
    grid = np.linspace(0.0, 3.0, 40)

    surface = true_survival(dgp, grid, result.truth, params)

    assert surface.shape == (200, grid.size)
    np.testing.assert_allclose(surface[:, 0], 1.0, atol=1e-12)
    assert np.all(surface >= 0.0) and np.all(surface <= 1.0)
    assert np.all(np.diff(surface, axis=1) <= 1e-12), f"{dgp} survival increases"


def test_cphm_matches_the_closed_form_exactly() -> None:
    """The one case simple enough to write down independently and compare."""
    params = CASES["cphm"]
    result = simulate("cphm", n=50, **params, seed=3)
    grid = np.array([0.0, 0.25, 1.0, 4.0])

    surface = true_survival("cphm", grid, result.truth, params)

    covariate = np.asarray(result.truth["covariates"], dtype=float).reshape(-1, 1)
    expected = np.exp(-grid.reshape(1, -1) * np.exp(params["beta"] * covariate))
    np.testing.assert_allclose(surface, expected, rtol=1e-12)


def test_mixture_cure_plateaus_at_the_cure_probability() -> None:
    """The property no proportional-hazards model can reproduce.

    Survival does not decay to zero: it flattens at pi(X), the probability the
    subject is cured. This is the structural feature that makes `mixture_cure`
    the sharpest test of the paper's question.
    """
    params = CASES["mixture_cure"]
    result = simulate("mixture_cure", n=500, **params, seed=11)

    far = true_survival("mixture_cure", np.array([1e4]), result.truth, params).ravel()

    cure_lp = np.asarray(result.truth["cure_linear_predictor"], dtype=float)
    logit = np.log(params["cure_fraction"] / (1 - params["cure_fraction"]))
    expected = 1.0 / (1.0 + np.exp(-(logit + cure_lp)))

    np.testing.assert_allclose(far, expected, atol=1e-10)
    assert far.min() > 0.01, "the plateau should be well away from zero"


def test_log_logistic_truth_matches_the_generator_clipping() -> None:
    """The truth function must match the implemented winsorised DGP."""
    params = CASES["aft_log_logistic"]
    result = simulate("aft_log_logistic", n=200, **params, seed=5)
    eta = np.asarray(result.truth["linear_predictor"], dtype=float)
    shape = float(params["shape"])
    scale = float(params["scale"])

    lower_time = scale * (0.001 / 0.999) ** (1.0 / shape) * np.exp(-eta / shape)
    upper_time = scale * (0.999 / 0.001) ** (1.0 / shape) * np.exp(-eta / shape)

    lower_surface = true_survival("aft_log_logistic", lower_time, result.truth, params)
    upper_surface = true_survival("aft_log_logistic", upper_time, result.truth, params)

    np.testing.assert_allclose(np.diag(lower_surface), 0.999, atol=1e-12)
    np.testing.assert_allclose(np.diag(upper_surface), 0.0, atol=1e-12)


def test_unsupported_dgp_raises_with_the_reason() -> None:
    with pytest.raises(KeyError, match="tdcm"):
        true_survival("tdcm", np.array([1.0]), {"linear_predictor": np.zeros(2)}, {})


def test_negative_times_are_rejected() -> None:
    params = CASES["cphm"]
    result = simulate("cphm", n=10, **params, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        true_survival("cphm", np.array([-1.0, 1.0]), result.truth, params)
