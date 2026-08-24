"""Correctness tests for the time-dependent covariate generator.

`gen_tdcm` switches a covariate on at a crossover time drawn jointly with the
baseline covariate. These check that the switch does what it claims: that the
hazard afterwards is ``exp(beta[1])`` times the hazard before, that the
recorded ``tdcov`` describes the interval actually observed, and that the times
are times.

The defect they guard against shipped through 2.0.2. The inversion for a draw
falling after the crossover carried ``x * (1 - exp(beta[1]))`` where the algebra
gives ``x * (exp(beta[1]) - 1)``, so those draws landed *before* the crossover
and, for a large enough ``beta[1]``, went negative.

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_surv import generate, simulate

_N = 40_000
_SEED = 20260824

# Effectively no censoring, so every latent event time is observed.
_NO_CENSORING = 1e6

_DIST_PAR = [1.0, 2.0, 1.0, 2.0]


def _simulate(beta1: float, *, beta0: float = 0.0, lam: float = 1.0, n: int = _N):
    return simulate(
        "tdcm",
        n=n,
        dist="weibull",
        corr=0.5,
        dist_par=_DIST_PAR,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        beta=[beta0, beta1],
        lam=lam,
        seed=_SEED,
    )


def _hazard_either_side(result) -> tuple[float, float]:
    """Occurrence over exposure, before and after the crossover."""
    crossover = np.asarray(result.truth["crossover_time"])
    stop = result.data["stop"].to_numpy()
    status = result.data["status"].to_numpy()
    tdcov = result.data["tdcov"].to_numpy()

    before = (
        int(((tdcov == 0) & (status == 1)).sum()) / np.minimum(stop, crossover).sum()
    )
    after = (
        int(((tdcov == 1) & (status == 1)).sum())
        / np.maximum(stop - crossover, 0).sum()
    )
    return float(before), float(after)


# --------------------------------------------------------------------------
# Times are times
# --------------------------------------------------------------------------


@pytest.mark.parametrize("beta1", [0.3, 1.0, 2.0])
def test_survival_times_are_never_negative(beta1: float) -> None:
    """The headline symptom: 2.0.2 produced negative times for large beta[1]."""
    frame = generate(
        model="tdcm",
        n=_N,
        dist="weibull",
        corr=0.5,
        dist_par=_DIST_PAR,
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        beta=[0.5, beta1],
        lam=1.0,
        seed=_SEED,
    )

    assert frame["stop"].min() >= 0.0
    assert (frame["stop"] > frame["start"]).all()


# --------------------------------------------------------------------------
# The switch multiplies the hazard by exp(beta[1])
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beta1", [0.0, np.log(2.0), np.log(0.5), 1.0], ids=["none", "double", "halve", "e"]
)
def test_hazard_after_the_crossover_is_exp_beta_times_the_hazard_before(
    beta1: float,
) -> None:
    """With no baseline covariate effect, the ratio is exactly ``exp(beta[1])``.

    On the previous implementation this came out at 4.58 where it should have
    been 2.0.
    """
    before, after = _hazard_either_side(_simulate(beta1))

    np.testing.assert_allclose(after / before, np.exp(beta1), rtol=0.05)


@pytest.mark.parametrize("beta1", [0.0, 1.0, 2.0])
def test_hazard_before_the_crossover_is_the_baseline_rate(beta1: float) -> None:
    """Before the switch the hazard is ``lam``, untouched by ``beta[1]``.

    The previous implementation failed this too: its error fed back into the
    pre-crossover exposure, so the rate there read 1.94 against a ``lam`` of 1.
    """
    before, _ = _hazard_either_side(_simulate(beta1, lam=0.8))

    np.testing.assert_allclose(before, 0.8, rtol=0.05)


# --------------------------------------------------------------------------
# tdcov describes the interval that was observed
# --------------------------------------------------------------------------


def test_tdcov_is_exactly_whether_the_crossover_was_reached() -> None:
    result = _simulate(0.3, beta0=0.5)

    crossover = np.asarray(result.truth["crossover_time"])
    stop = result.data["stop"].to_numpy()
    tdcov = result.data["tdcov"].to_numpy()

    np.testing.assert_array_equal(tdcov == 1.0, crossover <= stop)


def test_a_subject_censored_before_its_crossover_has_tdcov_zero() -> None:
    """Its covariate never switched during the time it was observed."""
    result = simulate(
        "tdcm",
        n=_N,
        dist="weibull",
        corr=0.5,
        dist_par=_DIST_PAR,
        model_cens="uniform",
        cens_par=0.5,  # heavy censoring, so many exits precede the crossover
        beta=[0.5, 0.3],
        lam=1.0,
        seed=_SEED,
    )

    crossover = np.asarray(result.truth["crossover_time"])
    stop = result.data["stop"].to_numpy()
    early_exit = stop < crossover

    assert early_exit.sum() > 0, "expected some exits before the crossover"
    assert (result.data["tdcov"].to_numpy()[early_exit] == 0.0).all()


def test_events_recorded_as_switched_happen_after_the_crossover() -> None:
    result = _simulate(0.3, beta0=0.5)

    crossover = np.asarray(result.truth["crossover_time"])
    stop = result.data["stop"].to_numpy()
    switched = result.data["tdcov"].to_numpy() == 1.0

    assert (stop[switched] >= crossover[switched]).all()


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------


def test_crossover_times_are_positive_and_reported() -> None:
    result = _simulate(0.3, n=1000)
    crossover = np.asarray(result.truth["crossover_time"])

    assert crossover.shape == (1000,)
    assert (crossover > 0).all()


def test_beta_of_length_three_still_warns_rather_than_failing() -> None:
    """The deprecated third coefficient is unaffected by this change."""
    with pytest.warns(DeprecationWarning):
        generate(
            model="tdcm",
            n=10,
            dist="weibull",
            corr=0.5,
            dist_par=_DIST_PAR,
            model_cens="uniform",
            cens_par=5.0,
            beta=[0.5, 0.3, 0.2],
            lam=1.0,
            seed=_SEED,
        )
