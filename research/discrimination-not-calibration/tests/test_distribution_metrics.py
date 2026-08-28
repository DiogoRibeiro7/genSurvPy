"""D-calibration and Antolini concordance, the two measures defined on the
predicted *distribution* rather than on a scalar score.

Both were added when the study's contribution was narrowed. Once the claim is
about how far an estimated individual survival distribution has moved from the
truth, it is not defensible to assess calibration only at one horizon, nor to
assess discrimination on a score obtained by collapsing the distribution --
which Sonabend et al. (2022) show can be made to look better or worse depending
on how the collapsing is done.

The tests below construct cases whose answer is known in advance, rather than
comparing against whatever the implementation happened to produce first.
"""

from __future__ import annotations

import numpy as np
import pytest
from survival_misspec.metrics import antolini_concordance, d_calibration


def _exponential_surface(rates: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.exp(-np.outer(rates, grid))


def test_d_calibration_does_not_reject_a_correct_model() -> None:
    """The case that matters most, and the one that caught the first bug.

    Data drawn from exactly the predicted distribution must not be rejected. An
    earlier version clipped every subject surviving past the grid to S(tau),
    which piled them into one bin and rejected Cox on a correctly specified Cox
    mechanism at p < 1e-4 despite a MISE of 6e-5.
    """
    rng = np.random.default_rng(0)
    n = 4000
    rates = rng.uniform(0.5, 2.0, size=n)
    times = rng.exponential(1.0 / rates)

    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 60)
    predicted = _exponential_surface(rates, grid)

    outcome = d_calibration(predicted, grid, times, np.ones(n, dtype=bool))

    assert outcome["d_calibration_p"] > 0.01, (
        f"a model predicting the exact generating distribution was rejected "
        f"(p={outcome['d_calibration_p']:.4g})"
    )


def test_d_calibration_rejects_a_systematically_wrong_model() -> None:
    """A metric that never rejects is not a metric."""
    rng = np.random.default_rng(1)
    n = 4000
    rates = rng.uniform(0.5, 2.0, size=n)
    times = rng.exponential(1.0 / rates)

    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 60)
    # Predicts a hazard half the truth, so survival is systematically too high.
    predicted = _exponential_surface(rates * 0.5, grid)

    outcome = d_calibration(predicted, grid, times, np.ones(n, dtype=bool))

    assert (
        outcome["d_calibration_p"] < 1e-3
    ), "a model predicting half the true hazard was not rejected"


def test_d_calibration_uses_censored_subjects_rather_than_dropping_them() -> None:
    """Discarding censored observations biases towards early failures.

    A censored subject is not uninformative: it is known to have survived at
    least to its censoring time, which constrains where its transformed value
    can lie. Under heavy censoring, dropping them would leave the statistic
    computed on exactly the subjects a censored study over-observes.
    """
    rng = np.random.default_rng(2)
    n = 4000
    rates = rng.uniform(0.5, 2.0, size=n)
    event_time = rng.exponential(1.0 / rates)
    censor_time = rng.exponential(1.0, size=n)

    observed = np.minimum(event_time, censor_time)
    event = event_time <= censor_time
    assert 0.2 < 1 - event.mean() < 0.8, "the fixture should be genuinely censored"

    grid = np.linspace(0.0, float(np.quantile(observed, 0.95)), 60)
    predicted = _exponential_surface(rates, grid)

    outcome = d_calibration(predicted, grid, observed, event)

    assert outcome["d_calibration_p"] > 0.01, (
        f"the correct model was rejected under censoring "
        f"(p={outcome['d_calibration_p']:.4g}); the censored contributions are "
        f"probably being mishandled"
    )


def test_antolini_agrees_with_a_known_ranking_when_hazards_are_proportional() -> None:
    """Under proportional hazards the ranking does not depend on the horizon.

    Antolini's index should then be high and close to what any correct ranking
    achieves, because comparing at each event time and comparing at a fixed
    horizon give the same ordering.
    """
    rng = np.random.default_rng(3)
    n = 1500
    rates = rng.uniform(0.3, 3.0, size=n)
    times = rng.exponential(1.0 / rates)

    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 60)
    predicted = _exponential_surface(rates, grid)

    outcome = antolini_concordance(
        predicted, grid, times, np.ones(n, dtype=bool), max_events=400
    )

    assert outcome["antolini_pairs"] > 0
    assert outcome["c_index_antolini"] > 0.65, (
        f"a model given the exact hazards should rank well; got "
        f"{outcome['c_index_antolini']:.4f}"
    )


def test_antolini_is_uninformative_for_a_model_that_cannot_discriminate() -> None:
    """Identical predictions for everyone must give one half, not an accident."""
    rng = np.random.default_rng(4)
    n = 800
    times = rng.exponential(1.0, size=n)
    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 40)
    predicted = np.tile(np.exp(-grid), (n, 1))

    outcome = antolini_concordance(predicted, grid, times, np.ones(n, dtype=bool))

    assert outcome["c_index_antolini"] == pytest.approx(0.5, abs=1e-9)


def test_antolini_handles_a_sample_with_no_events() -> None:
    grid = np.linspace(0.0, 1.0, 10)
    predicted = np.tile(np.exp(-grid), (5, 1))
    outcome = antolini_concordance(
        predicted, grid, np.full(5, 0.5), np.zeros(5, dtype=bool)
    )
    assert np.isnan(outcome["c_index_antolini"])
    assert outcome["antolini_pairs"] == 0


def test_antolini_subsampling_is_deterministic() -> None:
    """The subsample must not make the metric depend on when it is computed."""
    rng = np.random.default_rng(5)
    n = 2000
    rates = rng.uniform(0.3, 3.0, size=n)
    times = rng.exponential(1.0 / rates)
    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 40)
    predicted = _exponential_surface(rates, grid)

    first = antolini_concordance(
        predicted, grid, times, np.ones(n, bool), max_events=200
    )
    second = antolini_concordance(
        predicted, grid, times, np.ones(n, bool), max_events=200
    )

    assert first == second
