"""Statistical correctness tests for the simulators.

These assert that the generators reproduce the distributions they claim, rather
than only the shape and dtypes of the returned frame. Every test here maps to a
defect that shipped in 1.2.0 and that the existing shape-based tests could not
detect.

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from gen_surv.bivariate import sample_bivariate_distribution
from gen_surv.competing_risks import gen_competing_risks
from gen_surv.tdcm import gen_tdcm
from gen_surv.thmm import gen_thmm
from gen_surv.validation import LengthError

# Large enough to detect a wrong distribution, small enough to keep CI quick.
_N = 200_000
_SEED = 20260810


# --------------------------------------------------------------------------
# Bivariate sampler: marginals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.5, 1.0, 2.5])
def test_bivariate_exponential_marginals_are_exponential(lam: float) -> None:
    """Both marginals must be Exponential(lam), whatever the dependence."""
    sample = sample_bivariate_distribution(
        _N, "exponential", 0.4, [lam, lam], seed=_SEED
    )

    for column in range(2):
        result = stats.kstest(sample[:, column], "expon", args=(0.0, 1.0 / lam))
        assert result.pvalue > 0.01, f"column {column} is not Exponential({lam})"


def test_bivariate_exponential_moments_match_theory() -> None:
    """Mean and variance must match 1/lam and 1/lam**2."""
    lam = 2.0
    sample = sample_bivariate_distribution(
        _N, "exponential", 0.0, [lam, lam], seed=_SEED
    )

    assert sample[:, 0].mean() == pytest.approx(1.0 / lam, rel=0.02)
    assert sample[:, 0].var() == pytest.approx(1.0 / lam**2, rel=0.05)


def test_bivariate_exponential_is_not_chi_squared() -> None:
    """Guard against the 1.2.0 transform, which returned chi-squared draws.

    Releases up to 1.2.0 used ``u = 1 - exp(-z**2 / 2)``, so the "exponential"
    marginal was really ``chi2(1) / 2`` with mean 0.5 instead of 1.
    """
    sample = sample_bivariate_distribution(
        _N, "exponential", 0.0, [1.0, 1.0], seed=_SEED
    )

    assert sample[:, 0].mean() > 0.9, "mean near 0.5 means the old transform is back"

    as_chi2 = stats.kstest(sample[:, 0] * 2.0, lambda x: stats.chi2.cdf(x, df=1))
    assert as_chi2.pvalue < 1e-6, "draws still look chi-squared"


def test_bivariate_weibull_marginals_are_weibull() -> None:
    """Weibull marginals must match the requested shape and scale.

    With ``dist_par = [a, b, a, b]`` the sampler inverts ``F(x) = 1 - exp(-a
    x**b)``, which is a Weibull with shape ``b`` and scale ``a ** (-1 / b)``.
    """
    a, b = 1.5, 2.0
    sample = sample_bivariate_distribution(_N, "weibull", 0.5, [a, b, a, b], seed=_SEED)

    scale = a ** (-1.0 / b)
    result = stats.kstest(sample[:, 0], "weibull_min", args=(b, 0.0, scale))
    assert result.pvalue > 0.01


# --------------------------------------------------------------------------
# Bivariate sampler: dependence
# --------------------------------------------------------------------------


@pytest.mark.parametrize("corr", [-0.8, -0.4, 0.4, 0.8])
def test_bivariate_preserves_sign_of_dependence(corr: float) -> None:
    """Negative ``corr`` must give negative dependence.

    The 1.2.0 transform squared the normals, mapping ``+r`` and ``-r`` onto the
    same positive dependence and making negative dependence unreachable.
    """
    sample = sample_bivariate_distribution(
        _N, "exponential", corr, [1.0, 1.0], seed=_SEED
    )
    realized = float(np.corrcoef(sample.T)[0, 1])

    assert np.sign(realized) == np.sign(corr), f"corr={corr} gave {realized:+.4f}"


def test_bivariate_dependence_is_monotone_in_corr() -> None:
    """Realized dependence must increase with the requested correlation."""
    realized = [
        float(
            np.corrcoef(
                sample_bivariate_distribution(
                    _N, "exponential", corr, [1.0, 1.0], seed=_SEED
                ).T
            )[0, 1]
        )
        for corr in (-0.8, -0.4, 0.0, 0.4, 0.8)
    ]

    assert realized == sorted(realized), f"not monotone: {realized}"


def test_bivariate_rank_correlation_matches_gaussian_copula() -> None:
    """Spearman correlation must match the Gaussian copula identity.

    For a Gaussian copula with parameter ``rho``, Spearman's rho is
    ``(6 / pi) * arcsin(rho / 2)`` regardless of the marginals, so this pins the
    dependence structure independently of the marginal transforms.
    """
    rho = 0.6
    sample = sample_bivariate_distribution(
        _N, "exponential", rho, [1.0, 2.0], seed=_SEED
    )

    expected = 6.0 / np.pi * np.arcsin(rho / 2.0)
    observed = float(stats.spearmanr(sample[:, 0], sample[:, 1]).statistic)

    assert observed == pytest.approx(expected, abs=0.01)


# --------------------------------------------------------------------------
# Competing risks: no fabricated events
# --------------------------------------------------------------------------


def test_competing_risks_does_not_fabricate_events_under_heavy_censoring() -> None:
    """An all-censored sample must stay all-censored.

    Releases up to 1.2.0 overwrote ``status[0]`` and ``status[1]`` whenever fewer
    than two distinct statuses appeared, attaching event labels to subjects whose
    event times had not occurred.
    """
    frame = gen_competing_risks(
        n=6,
        n_risks=2,
        baseline_hazards=[1e-6, 1e-6],
        max_time=None,
        model_cens="uniform",
        cens_par=1e-3,
        seed=_SEED,
    )

    assert set(frame["status"].unique()) == {0}, "events were fabricated"


def test_competing_risks_allows_a_single_observed_cause() -> None:
    """One absent cause is a valid stochastic outcome, not an error."""
    frame = gen_competing_risks(
        n=8,
        n_risks=2,
        baseline_hazards=[5.0, 1e-9],
        max_time=None,
        model_cens="uniform",
        cens_par=100.0,
        seed=_SEED,
    )

    assert 2 not in set(frame["status"].unique()), "an impossible cause appeared"


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def _bivariate(seed: object) -> np.ndarray:
    return sample_bivariate_distribution(32, "exponential", 0.3, [1.0, 1.0], seed=seed)


def _tdcm(seed: object) -> np.ndarray:
    return gen_tdcm(
        n=32,
        dist="exponential",
        corr=0.3,
        dist_par=[0.5, 1.0],
        model_cens="uniform",
        cens_par=2.0,
        beta=[0.1, 0.2],
        lam=0.5,
        seed=seed,
    ).to_numpy()


def _thmm(seed: object) -> np.ndarray:
    return gen_thmm(
        n=32,
        model_cens="uniform",
        cens_par=5.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.1, 0.1, 0.2],
        seed=seed,
    ).to_numpy()


_GENERATORS = {"bivariate": _bivariate, "tdcm": _tdcm, "thmm": _thmm}


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_equal_seeds_give_equal_draws(name: str) -> None:
    """These three drew from the global NumPy state before 1.3.0."""
    generator = _GENERATORS[name]
    assert np.allclose(generator(123), generator(123))


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_different_seeds_give_different_draws(name: str) -> None:
    """A seed must actually reach the draws rather than being ignored."""
    generator = _GENERATORS[name]
    assert not np.allclose(generator(123), generator(456))


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_draws_ignore_the_global_numpy_state(name: str) -> None:
    """Seeding the legacy global state must not change a seeded draw."""
    generator = _GENERATORS[name]

    np.random.seed(0)
    first = generator(2024)
    np.random.seed(999)
    second = generator(2024)

    assert np.allclose(first, second)


@pytest.mark.parametrize("name", sorted(_GENERATORS))
def test_a_shared_generator_is_accepted(name: str) -> None:
    """Passing a Generator lets callers share one stream across simulators."""
    generator = _GENERATORS[name]
    assert np.allclose(
        generator(np.random.default_rng(7)), generator(np.random.default_rng(7))
    )


# --------------------------------------------------------------------------
# TDCM coefficient contract
# --------------------------------------------------------------------------


def test_tdcm_accepts_the_documented_two_coefficients() -> None:
    """The documented length-2 beta must be accepted without warning.

    Releases up to 1.2.0 required three coefficients and ignored the third, so
    the documented call -- and the function's own docstring example -- raised.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = _tdcm(_SEED)

    assert len(frame) == 32


def test_tdcm_three_coefficients_still_work_but_warn() -> None:
    """The old length-3 call keeps working, with a deprecation warning."""
    with pytest.warns(DeprecationWarning, match="two coefficients"):
        gen_tdcm(
            n=8,
            dist="exponential",
            corr=0.3,
            dist_par=[0.5, 1.0],
            model_cens="uniform",
            cens_par=2.0,
            beta=[0.1, 0.2, 0.9],
            lam=0.5,
            seed=_SEED,
        )


def test_tdcm_rejects_other_beta_lengths() -> None:
    """Lengths other than 2 or 3 remain errors."""
    with pytest.raises(LengthError):
        gen_tdcm(
            n=8,
            dist="exponential",
            corr=0.3,
            dist_par=[0.5, 1.0],
            model_cens="uniform",
            cens_par=2.0,
            beta=[0.1],
            lam=0.5,
            seed=_SEED,
        )
