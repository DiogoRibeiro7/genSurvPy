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
from survival_misspec.metrics import (
    antolini_concordance,
    d_calibration,
    prediction_error,
    truth_recovery,
)


def _exponential_surface(rates: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.exp(-np.outer(rates, grid))


class _FakeStep:
    def __init__(self, x: list[float], y: list[float]) -> None:
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)

    def __call__(self, time: float) -> float:
        index = int(np.searchsorted(self.x, time, side="left"))
        index = min(max(index, 0), len(self.y) - 1)
        return float(self.y[index])


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


def test_d_calibration_uses_step_values_not_linear_interpolation() -> None:
    """Distribution metrics must not invent between-step probabilities."""
    grid = np.array([0.0, 1.0, 2.0])
    predicted = np.array([[1.0, 0.8, 0.2]])
    observed = np.array([0.5])

    outcome = d_calibration(predicted, grid, observed, np.ones(1, dtype=bool))

    # Right-continuous step lookup uses S(1.0) = 0.8. Linear interpolation would
    # use 0.9 and place the event in the top bin.
    expected = np.zeros(10)
    expected[8] = 1.0
    uniform = np.full(10, 0.1)
    statistic = float(((expected - uniform) ** 2 / uniform).sum())
    assert outcome["d_calibration_statistic"] == pytest.approx(statistic)


def test_d_calibration_can_use_native_step_functions() -> None:
    grid = np.array([0.0, 1.0, 2.0])
    predicted = np.array([[1.0, 0.2, 0.2]])
    observed = np.array([0.5])
    functions = [_FakeStep([0.0, 0.5, 2.0], [1.0, 0.8, 0.2])]

    outcome = d_calibration(
        predicted,
        grid,
        observed,
        np.ones(1, dtype=bool),
        survival_functions=functions,
    )

    expected = np.zeros(10)
    expected[8] = 1.0
    uniform = np.full(10, 0.1)
    statistic = float(((expected - uniform) ** 2 / uniform).sum())
    assert outcome["d_calibration_statistic"] == pytest.approx(statistic)


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

    correct = antolini_concordance(
        predicted, grid, times, np.ones(n, dtype=bool), max_events=400
    )
    # The same curves handed to the wrong subjects. A threshold on the index
    # alone would only pin whatever the code currently returns; this pins the
    # property that actually matters.
    shuffled = antolini_concordance(
        predicted[np.random.default_rng(0).permutation(n)],
        grid,
        times,
        np.ones(n, dtype=bool),
        max_events=400,
    )

    assert correct["antolini_pairs"] > 0
    assert correct["c_index_antolini"] > shuffled["c_index_antolini"] + 0.1, (
        f"a model given the exact hazards should rank far better than the same "
        f"curves misassigned; got {correct['c_index_antolini']:.4f} against "
        f"{shuffled['c_index_antolini']:.4f}"
    )


def test_antolini_scores_identical_predictions_by_the_published_rule() -> None:
    """Equation 12 is a strict inequality, so a tie is not half a concordance.

    Harrell's convention gives ties one half; Antolini's estimator does not.
    With identical predictions for everyone the index is therefore 0, not 0.5.
    That looks harsh, and it is the published definition -- so the tie fraction
    is reported alongside, because two of this study's estimators predict step
    functions whose values are frequently exactly equal and the choice of
    convention would otherwise silently decide their score.
    """
    rng = np.random.default_rng(4)
    n = 800
    times = rng.exponential(1.0, size=n)
    grid = np.linspace(0.0, float(np.quantile(times, 0.95)), 40)
    predicted = np.tile(np.exp(-grid), (n, 1))

    outcome = antolini_concordance(predicted, grid, times, np.ones(n, dtype=bool))

    assert outcome["c_index_antolini"] == pytest.approx(0.0, abs=1e-9)
    assert outcome["antolini_tie_fraction"] == pytest.approx(1.0, abs=1e-9)


def test_antolini_administratively_censors_at_the_horizon() -> None:
    """Section 2.3 restricts to [0, tau] by censoring at tau, not by clipping.

    An event after the horizon is not an event within it, so it cannot be the
    earlier member of a comparable pair. Clipping it to the last grid point --
    the previous behaviour -- invented comparisons the definition excludes and
    evaluated them at the wrong time.
    """
    n = 60
    rng = np.random.default_rng(9)
    rates = rng.uniform(0.5, 2.0, size=n)
    times = np.linspace(0.1, 10.0, n)
    grid = np.linspace(0.0, 2.0, 30)
    predicted = _exponential_surface(rates, grid)

    outcome = antolini_concordance(predicted, grid, times, np.ones(n, dtype=bool))

    # Only subjects failing at or before tau = 2.0 may act as events, and each
    # needs someone later, so the last one inside the horizon cannot contribute.
    inside = int((times <= 2.0).sum())
    assert outcome["antolini_pairs"] > 0
    assert outcome["antolini_pairs"] <= inside * n


def test_antolini_uses_the_next_step_at_event_time() -> None:
    grid = np.array([0.0, 1.0, 2.0])
    predicted = np.array(
        [
            [1.0, 0.4, 0.4],
            [1.0, 0.6, 0.3],
        ]
    )
    time = np.array([0.5, 1.5])
    event = np.array([True, True])

    outcome = antolini_concordance(predicted, grid, time, event)

    assert outcome["antolini_pairs"] == 1
    assert outcome["c_index_antolini"] == pytest.approx(1.0)


def test_antolini_can_use_native_step_functions() -> None:
    grid = np.array([0.0, 1.0, 2.0])
    predicted = np.array(
        [
            [1.0, 0.9, 0.9],
            [1.0, 0.1, 0.1],
        ]
    )
    functions = [
        _FakeStep([0.0, 0.5, 2.0], [1.0, 0.2, 0.2]),
        _FakeStep([0.0, 0.5, 2.0], [1.0, 0.8, 0.8]),
    ]

    outcome = antolini_concordance(
        predicted,
        grid,
        np.array([0.5, 1.5]),
        np.array([True, True]),
        survival_functions=functions,
    )

    assert outcome["antolini_pairs"] == 1
    assert outcome["c_index_antolini"] == pytest.approx(1.0)


def test_truth_recovery_reports_horizon_normalised_squared_loss() -> None:
    grid = np.linspace(0.0, 2.0, 21)
    truth = np.tile(np.exp(-grid), (1, 1))
    predicted = truth + 0.2

    outcome = truth_recovery(predicted, truth, grid, tau=2.0)

    assert outcome["mise"] == pytest.approx(0.08)
    assert outcome["normalised_mise"] == pytest.approx(0.04)
    assert outcome["root_mean_integrated_squared_error"] == pytest.approx(0.2)


def test_prediction_error_only_labels_a_point_metric_when_it_is_at_tau() -> None:
    grid = np.array([0.0, 0.5, 1.0])
    predicted = np.tile(np.exp(-grid), (20, 1))
    time = np.linspace(0.1, 0.9, 20)
    event = np.ones(20, dtype=bool)

    outcome = prediction_error(predicted, grid, time, event, time, event, tau=1.0)

    assert np.isnan(outcome["brier_at_tau"])
    assert np.isnan(outcome["auc_at_tau"])
    assert np.isnan(outcome["brier_at_tau_time"])


def test_prediction_error_reports_tau_when_tau_is_inside_support() -> None:
    grid = np.array([0.0, 0.5, 1.0])
    predicted = np.tile(np.exp(-grid), (30, 1))
    time = np.linspace(0.1, 1.4, 30)
    event = np.ones(30, dtype=bool)

    outcome = prediction_error(predicted, grid, time, event, time, event, tau=1.0)

    assert outcome["brier_at_tau_time"] == pytest.approx(1.0)
    assert outcome["auc_at_tau_time"] == pytest.approx(1.0)
    assert np.isfinite(outcome["brier_at_tau"])


def test_prediction_error_does_not_shorten_a_fixed_interval() -> None:
    grid = np.array([0.0, 0.5, 1.0])
    predicted = np.tile(np.exp(-grid), (20, 1))
    time = np.linspace(0.1, 0.8, 20)
    event = np.ones(20, dtype=bool)

    outcome = prediction_error(
        predicted,
        grid,
        time,
        event,
        time,
        event,
        tau=1.0,
        evaluation_times=np.array([0.5, 1.0]),
    )

    assert np.isnan(outcome["integrated_brier_score"])
    assert np.isnan(outcome["auc_mean"])
    assert "fixed IPCW interval" in outcome["prediction_error_note"]


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


def test_expected_mortality_is_one_transformation_for_every_model() -> None:
    """Sonabend et al. (2022) call the alternative the third form of C-hacking.

    Scoring each model on its own native risk means comparing a Cox partial
    hazard against a negative expected survival time against a forest's summed
    cumulative hazard, all with one concordance measure — three different
    mathematical objects, so the comparison says little about the models. This
    derives the score from the predicted curve in one fixed way, so the only
    thing differing between estimators is the curve itself.
    """
    from survival_misspec.metrics import expected_mortality

    grid = np.linspace(0.0, 2.0, 40)
    rates = np.array([0.5, 1.0, 2.0])
    predicted = _exponential_surface(rates, grid)

    risk = expected_mortality(predicted, grid)

    # A higher hazard must give a higher risk, and the transformation is exact
    # for the exponential: the integral of H(t) = rate * t over [0, tau].
    assert risk[0] < risk[1] < risk[2]
    np.testing.assert_allclose(risk, rates * grid[-1] ** 2 / 2.0, rtol=1e-3)


def test_expected_mortality_ranks_a_survival_curve_the_right_way_round() -> None:
    """Higher risk must mean earlier failure, or every concordance flips."""
    from survival_misspec.metrics import expected_mortality

    grid = np.linspace(0.0, 1.0, 30)
    healthier = np.exp(-0.2 * grid).reshape(1, -1)
    sicker = np.exp(-3.0 * grid).reshape(1, -1)

    risk = expected_mortality(np.vstack([healthier, sicker]), grid)

    assert risk[1] > risk[0]
