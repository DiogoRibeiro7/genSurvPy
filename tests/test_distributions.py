"""Distribution tests for the generators that had none.

The roadmap's highest-priority item, and for a reason: the bivariate sampler
returned ``chi2(1)/2`` where an exponential was requested and survived many
releases, because every test checked the shape of the frame rather than the
distribution in it. Two more defects have been found since by asking
distributional questions -- a piecewise interval drawn at the wrong hazard, and
a sign error in the ``tdcm`` inversion that produced negative times.

Each test here applies the **probability integral transform**: rearrange the
sampled times by the model's own cumulative hazard and the result must be
Uniform(0, 1), or equivalently Exponential(1). That is a complete statement of
the distribution -- shape, scale and covariate effect at once -- rather than a
check on a moment or two.

The transforms use the latent event times reported by :func:`gen_surv.simulate`,
so censoring does not have to be worked around.

Coverage by generator:

===========================  =========================================
``cphm``                     here
``aft_ln``                   here
``aft_weibull``              here
``aft_log_logistic``         here
``competing_risks``          here
``competing_risks_weibull``  here
``mixture_cure``             here
``piecewise_exponential``    ``test_piecewise_hazards.py``
``recurrent_events``         ``test_recurrent.py``
``cmm``, ``thmm``, ``tdcm``  ``test_statistical_correctness.py``,
                             ``test_tdcm_crossover.py``
===========================  =========================================

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from gen_surv import simulate

_N = 40_000
_SEED = 3

# Effectively no censoring, so every latent time is drawn from the model alone.
_NO_CENSORING = 1e9

# A fixed seed makes these deterministic; the threshold is loose enough that a
# last-bit change in a library cannot trip it, and tight enough that a wrong
# distribution cannot pass. A correct generator gives p uniform on (0, 1).
_MIN_P = 0.01


def _assert_uniform(sample: np.ndarray, label: str) -> None:
    """The transformed sample must be indistinguishable from Uniform(0, 1)."""
    p = stats.kstest(sample, "uniform").pvalue
    assert p > _MIN_P, f"{label}: KS against Uniform(0,1) gave p={p:.5f}"


def _assert_unit_exponential(sample: np.ndarray, label: str) -> None:
    """The transformed sample must be indistinguishable from Exponential(1)."""
    p = stats.kstest(sample, "expon").pvalue
    assert p > _MIN_P, f"{label}: KS against Exponential(1) gave p={p:.5f}"


# --------------------------------------------------------------------------
# Cox proportional hazards
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [0.0, 0.7, -0.7])
def test_cphm_event_times_are_exponential_with_the_declared_hazard(
    beta: float,
) -> None:
    """``T | X`` is Exponential with rate ``exp(beta * X)``.

    Equivalently ``T * exp(beta * X)`` is Exponential(1), which tests the
    baseline and the covariate effect in one statement.
    """
    result = simulate(
        "cphm",
        n=_N,
        beta=beta,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    event = np.asarray(result.truth["event_time"])
    covariate = np.asarray(result.truth["covariates"])

    _assert_unit_exponential(event * np.exp(beta * covariate), f"cphm beta={beta}")


def test_cphm_covariate_is_uniform_on_its_range() -> None:
    result = simulate(
        "cphm",
        n=_N,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    covariate = np.asarray(result.truth["covariates"])
    p = stats.kstest(covariate, "uniform", args=(0, 2)).pvalue

    assert p > _MIN_P, f"covariate is not Uniform(0, 2): p={p:.5f}"


# --------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------


def test_uniform_censoring_times_are_uniform_on_zero_to_cens_par() -> None:
    result = simulate(
        "cphm",
        n=_N,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=3.0,
        seed=_SEED,
    )

    censoring = np.asarray(result.truth["censoring_time"])
    p = stats.kstest(censoring, "uniform", args=(0, 3)).pvalue

    assert p > _MIN_P, f"censoring is not Uniform(0, 3): p={p:.5f}"


def test_exponential_censoring_uses_cens_par_as_the_mean_not_the_rate() -> None:
    """``cens_par`` is documented as the mean, which is the easy thing to invert."""
    result = simulate(
        "cphm",
        n=_N,
        beta=0.5,
        covariate_range=2.0,
        model_cens="exponential",
        cens_par=3.0,
        seed=_SEED,
    )

    censoring = np.asarray(result.truth["censoring_time"])

    np.testing.assert_allclose(censoring.mean(), 3.0, rtol=0.02)
    _assert_unit_exponential(censoring / 3.0, "exponential censoring")


# --------------------------------------------------------------------------
# Accelerated failure time
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.5, 0.8, 1.5])
def test_aft_log_normal_residuals_are_standard_normal(sigma: float) -> None:
    """``(log T - X'beta) / sigma`` is standard normal by construction."""
    result = simulate(
        "aft_ln",
        n=_N,
        beta=[0.5, -0.3],
        sigma=sigma,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    residual = (
        np.log(np.asarray(result.truth["event_time"]))
        - np.asarray(result.truth["linear_predictor"])
    ) / sigma

    p = stats.kstest(residual, "norm").pvalue
    assert p > _MIN_P, f"aft_ln sigma={sigma}: residuals are not N(0,1), p={p:.5f}"


@pytest.mark.parametrize("shape", [0.7, 1.4, 2.5])
def test_aft_weibull_cumulative_hazard_is_unit_exponential(shape: float) -> None:
    """``(T / scale) ** shape * exp(eta)`` is the integrated hazard."""
    scale = 1.1
    result = simulate(
        "aft_weibull",
        n=_N,
        beta=[0.5, -0.3],
        shape=shape,
        scale=scale,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    cumulative = (np.asarray(result.truth["event_time"]) / scale) ** shape * np.exp(
        np.asarray(result.truth["linear_predictor"])
    )

    _assert_unit_exponential(cumulative, f"aft_weibull shape={shape}")


@pytest.mark.parametrize("shape", [0.9, 1.3, 2.0])
def test_aft_log_logistic_odds_transform_is_uniform(shape: float) -> None:
    """With ``W = (T/scale) ** shape * exp(eta)``, ``W / (1 + W)`` is uniform."""
    scale = 1.7
    result = simulate(
        "aft_log_logistic",
        n=_N,
        beta=[0.5, -0.3],
        shape=shape,
        scale=scale,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    odds = (np.asarray(result.truth["event_time"]) / scale) ** shape * np.exp(
        np.asarray(result.truth["linear_predictor"])
    )

    _assert_uniform(odds / (1.0 + odds), f"aft_log_logistic shape={shape}")


def test_aft_weibull_shape_controls_the_direction_of_the_hazard() -> None:
    """Falling below 1, rising above it: the reason to choose the family."""
    times = {}
    for shape in (0.5, 2.0):
        result = simulate(
            "aft_weibull",
            n=_N,
            beta=[0.0, 0.0],
            shape=shape,
            scale=1.0,
            model_cens="uniform",
            cens_par=_NO_CENSORING,
            seed=_SEED,
        )
        times[shape] = np.asarray(result.truth["event_time"])

    # A falling hazard piles mass at short times and keeps a long tail.
    assert np.median(times[0.5]) < np.median(times[2.0])
    assert times[0.5].max() > times[2.0].max()


# --------------------------------------------------------------------------
# Competing risks
# --------------------------------------------------------------------------


def test_competing_risks_latent_times_are_exponential_per_cause() -> None:
    """Each cause's latent time is Exponential with its own hazard."""
    baseline = [0.4, 0.2]
    betas = [[0.8, 0.0], [0.0, -0.5]]
    result = simulate(
        "competing_risks",
        n=_N,
        n_risks=2,
        baseline_hazards=baseline,
        betas=betas,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    cause_times = np.asarray(result.truth["cause_times"])
    covariates = np.asarray(result.truth["covariates"])

    for cause, (hazard, coefficients) in enumerate(zip(baseline, betas)):
        rate = hazard * np.exp(covariates @ np.asarray(coefficients))
        _assert_unit_exponential(cause_times[:, cause] * rate, f"cause {cause + 1}")


def test_competing_risks_weibull_latent_times_are_weibull_per_cause() -> None:
    shapes, scales = [1.2, 0.8], [2.0, 1.5]
    betas = [[0.7, 0.0], [0.0, -0.4]]
    result = simulate(
        "competing_risks_weibull",
        n=_N,
        n_risks=2,
        shape_params=shapes,
        scale_params=scales,
        betas=betas,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    cause_times = np.asarray(result.truth["cause_times"])
    covariates = np.asarray(result.truth["covariates"])

    for cause in range(2):
        cumulative = (cause_times[:, cause] / scales[cause]) ** shapes[cause] * np.exp(
            covariates @ np.asarray(betas[cause])
        )
        _assert_unit_exponential(cumulative, f"weibull cause {cause + 1}")


def test_competing_risks_weibull_marginals_match_the_named_distribution() -> None:
    """Without covariate effects, each latent time is exactly Weibull."""
    shapes, scales = [1.2, 0.8], [2.0, 1.5]
    result = simulate(
        "competing_risks_weibull",
        n=_N,
        n_risks=2,
        shape_params=shapes,
        scale_params=scales,
        betas=[[0.0, 0.0], [0.0, 0.0]],
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    cause_times = np.asarray(result.truth["cause_times"])

    for cause in range(2):
        p = stats.kstest(
            cause_times[:, cause],
            "weibull_min",
            args=(shapes[cause], 0, scales[cause]),
        ).pvalue
        assert p > _MIN_P, f"cause {cause + 1} is not Weibull: p={p:.5f}"


def test_competing_risks_cause_shares_follow_the_hazards() -> None:
    """With constant hazards and no covariates, cause k wins with probability
    ``hazard_k / sum(hazards)``."""
    baseline = [0.6, 0.2, 0.2]
    result = simulate(
        "competing_risks",
        n=_N,
        n_risks=3,
        baseline_hazards=baseline,
        betas=[[0.0, 0.0]] * 3,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    observed = result.data["status"].value_counts(normalize=True).sort_index()
    expected = np.asarray(baseline) / sum(baseline)

    np.testing.assert_allclose(observed.loc[[1, 2, 3]].to_numpy(), expected, rtol=0.05)


# --------------------------------------------------------------------------
# Mixture cure
# --------------------------------------------------------------------------


@pytest.mark.parametrize("baseline_hazard", [0.5, 1.0, 2.0])
def test_mixture_cure_uncured_times_are_exponential(baseline_hazard: float) -> None:
    """The uncured fail at ``baseline_hazard * exp(eta)``; the cured never do."""
    result = simulate(
        "mixture_cure",
        n=_N,
        cure_fraction=0.3,
        baseline_hazard=baseline_hazard,
        betas_survival=[0.6, -0.3],
        betas_cure=[0.0, 0.0],
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    cured = np.asarray(result.truth["cured"]).astype(bool)
    event = np.asarray(result.truth["event_time"])[~cured]
    linear = np.asarray(result.truth["linear_predictor"])[~cured]

    _assert_unit_exponential(
        event * baseline_hazard * np.exp(linear), f"uncured, h0={baseline_hazard}"
    )


@pytest.mark.parametrize("target", [0.2, 0.4, 0.6])
def test_mixture_cure_fraction_matches_when_it_does_not_depend_on_covariates(
    target: float,
) -> None:
    result = simulate(
        "mixture_cure",
        n=_N,
        cure_fraction=target,
        baseline_hazard=1.0,
        betas_survival=[0.0, 0.0],
        betas_cure=[0.0, 0.0],
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    np.testing.assert_allclose(
        np.asarray(result.truth["cured"]).mean(), target, rtol=0.05
    )


def test_mixture_cure_probability_follows_its_logistic_model() -> None:
    """Cure status is Bernoulli with a logistic probability in the covariates.

    Checked by covariate decile, which would catch a link function applied in
    the wrong direction or on the wrong scale.
    """
    target = 0.4
    result = simulate(
        "mixture_cure",
        n=60_000,
        cure_fraction=target,
        baseline_hazard=1.0,
        betas_survival=[0.0, 0.0],
        betas_cure=[1.0, 0.0],
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        max_time=None,
        seed=_SEED,
    )

    cured = np.asarray(result.truth["cured"]).astype(float)
    linear = np.asarray(result.truth["cure_linear_predictor"])
    covariate = np.asarray(result.truth["covariates"])[:, 0]

    intercept = np.log(target / (1.0 - target))
    modelled = 1.0 / (1.0 + np.exp(-(intercept + linear)))

    edges = np.quantile(covariate, np.linspace(0, 1, 11))
    for i in range(10):
        upper = covariate <= edges[i + 1] if i == 9 else covariate < edges[i + 1]
        group = (covariate >= edges[i]) & upper
        np.testing.assert_allclose(
            cured[group].mean(), modelled[group].mean(), atol=0.02
        )


def test_mixture_cure_marks_the_cured_as_censored() -> None:
    """A cured subject can never have an observed event."""
    result = simulate(
        "mixture_cure",
        n=_N,
        cure_fraction=0.4,
        baseline_hazard=1.0,
        betas_survival=[0.0, 0.0],
        betas_cure=[0.0, 0.0],
        model_cens="uniform",
        cens_par=5.0,
        max_time=10.0,
        seed=_SEED,
    )

    cured = result.data["cured"].to_numpy().astype(bool)
    assert (result.data["status"].to_numpy()[cured] == 0).all()
