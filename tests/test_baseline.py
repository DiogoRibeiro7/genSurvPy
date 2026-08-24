"""Tests for the baseline hazard families.

Each family is checked three ways: against its closed form, against a numerical
derivative of its own cumulative hazard, and for the round trip that sampling
depends on -- ``H0_inverse(H0(t)) == t``. If that round trip is wrong the times
a generator draws are wrong, and nothing about the shape of the frame would
show it.
"""

from __future__ import annotations

import numpy as np
import pytest

from gen_surv.baseline import (
    BASELINES,
    BaselineHazard,
    ExponentialBaseline,
    GompertzBaseline,
    LogLogisticBaseline,
    PiecewiseConstantBaseline,
    WeibullBaseline,
)
from gen_surv.validation import ValidationError

# Times away from any breakpoint, so a central difference is valid everywhere.
GRID = np.array([0.01, 0.1, 0.37, 0.8, 1.4, 2.6, 4.2])

FAMILIES = [
    ExponentialBaseline(rate=0.7),
    WeibullBaseline(shape=1.8, scale=2.0),
    WeibullBaseline(shape=0.6, scale=1.0),
    GompertzBaseline(rate=0.4, shape=0.3),
    GompertzBaseline(rate=0.4, shape=-0.3),
    LogLogisticBaseline(shape=1.6, scale=1.2),
    PiecewiseConstantBaseline(breakpoints=[1.0, 3.0], hazard_rates=[0.5, 2.0, 0.2]),
]

IDS = [
    "exponential",
    "weibull-rising",
    "weibull-falling",
    "gompertz-rising",
    "gompertz-falling",
    "log-logistic",
    "piecewise",
]


@pytest.mark.parametrize("baseline", FAMILIES, ids=IDS)
def test_satisfies_the_protocol(baseline: BaselineHazard) -> None:
    assert isinstance(baseline, BaselineHazard)


@pytest.mark.parametrize("baseline", FAMILIES, ids=IDS)
def test_inverse_round_trips(baseline: BaselineHazard) -> None:
    """``H0_inverse(H0(t))`` must return ``t``: this is what sampling relies on."""
    values = np.asarray(baseline.cumulative_hazard(GRID))
    recovered = np.asarray(baseline.inverse_cumulative_hazard(values))

    np.testing.assert_allclose(recovered, GRID, rtol=1e-10)


@pytest.mark.parametrize("baseline", FAMILIES, ids=IDS)
def test_cumulative_hazard_starts_at_zero_and_increases(
    baseline: BaselineHazard,
) -> None:
    assert baseline.cumulative_hazard(0.0) == pytest.approx(0.0)
    values = np.asarray(baseline.cumulative_hazard(GRID))
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize("baseline", FAMILIES, ids=IDS)
def test_hazard_is_the_derivative_of_the_cumulative_hazard(
    baseline: BaselineHazard,
) -> None:
    eps = 1e-6
    numeric = (
        np.asarray(baseline.cumulative_hazard(GRID + eps))
        - np.asarray(baseline.cumulative_hazard(GRID - eps))
    ) / (2 * eps)

    np.testing.assert_allclose(np.asarray(baseline.hazard(GRID)), numeric, rtol=1e-4)


@pytest.mark.parametrize("baseline", FAMILIES, ids=IDS)
def test_scalar_and_array_calls_agree(baseline: BaselineHazard) -> None:
    scalar = baseline.cumulative_hazard(1.4)
    array = np.asarray(baseline.cumulative_hazard(np.array([1.4])))

    assert isinstance(scalar, float)
    assert scalar == pytest.approx(float(array[0]))


# --------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------


def test_exponential_closed_form() -> None:
    baseline = ExponentialBaseline(rate=0.7)

    np.testing.assert_allclose(np.asarray(baseline.cumulative_hazard(GRID)), 0.7 * GRID)
    np.testing.assert_allclose(np.asarray(baseline.hazard(GRID)), 0.7)


def test_weibull_closed_form() -> None:
    baseline = WeibullBaseline(shape=1.8, scale=2.0)

    np.testing.assert_allclose(
        np.asarray(baseline.cumulative_hazard(GRID)), (GRID / 2.0) ** 1.8
    )


def test_gompertz_closed_form() -> None:
    baseline = GompertzBaseline(rate=0.4, shape=0.3)

    expected = 0.4 / 0.3 * (np.exp(0.3 * GRID) - 1.0)
    np.testing.assert_allclose(np.asarray(baseline.cumulative_hazard(GRID)), expected)


def test_log_logistic_closed_form() -> None:
    baseline = LogLogisticBaseline(shape=1.6, scale=1.2)

    expected = np.log(1.0 + (GRID / 1.2) ** 1.6)
    np.testing.assert_allclose(np.asarray(baseline.cumulative_hazard(GRID)), expected)


def test_log_logistic_hazard_is_unimodal() -> None:
    """The defining feature: the hazard rises to a peak, then falls."""
    fine = np.linspace(0.01, 20.0, 4000)
    hazard = np.asarray(LogLogisticBaseline(shape=2.5, scale=1.0).hazard(fine))

    peak = int(np.argmax(hazard))
    assert 0 < peak < len(fine) - 1, "the peak must be interior"
    assert np.all(np.diff(hazard[:peak]) > 0)
    assert np.all(np.diff(hazard[peak:]) < 0)


def test_piecewise_matches_a_manual_walk() -> None:
    breakpoints, rates = [1.0, 3.0], [0.5, 2.0, 0.2]
    baseline = PiecewiseConstantBaseline(breakpoints, rates)

    # H0(4) = 0.5 * 1 + 2.0 * 2 + 0.2 * 1
    assert baseline.cumulative_hazard(4.0) == pytest.approx(0.5 + 4.0 + 0.2)
    assert baseline.cumulative_hazard(0.5) == pytest.approx(0.25)
    assert baseline.cumulative_hazard(2.0) == pytest.approx(0.5 + 2.0)

    # The rate at a breakpoint belongs to the interval it opens.
    assert baseline.hazard(0.999) == pytest.approx(0.5)
    assert baseline.hazard(1.0) == pytest.approx(2.0)
    assert baseline.hazard(3.0) == pytest.approx(0.2)


def test_piecewise_inverse_lands_in_the_right_interval() -> None:
    baseline = PiecewiseConstantBaseline([1.0, 3.0], [0.5, 2.0, 0.2])

    assert baseline.inverse_cumulative_hazard(0.25) == pytest.approx(0.5)
    assert baseline.inverse_cumulative_hazard(0.5 + 2.0) == pytest.approx(2.0)
    assert baseline.inverse_cumulative_hazard(0.5 + 4.0 + 0.2) == pytest.approx(4.0)


# --------------------------------------------------------------------------
# The declining Gompertz, whose total hazard is finite
# --------------------------------------------------------------------------


def test_declining_gompertz_has_a_finite_total_hazard() -> None:
    baseline = GompertzBaseline(rate=0.4, shape=-0.3)

    assert baseline.total_hazard == pytest.approx(0.4 / 0.3)
    assert baseline.cumulative_hazard(1e6) == pytest.approx(baseline.total_hazard)


def test_declining_gompertz_inverse_is_infinite_beyond_its_total() -> None:
    """Not an error: the event simply never happens."""
    baseline = GompertzBaseline(rate=0.4, shape=-0.3)

    assert baseline.inverse_cumulative_hazard(baseline.total_hazard + 1.0) == np.inf
    assert np.isfinite(baseline.inverse_cumulative_hazard(baseline.total_hazard * 0.9))


def test_rising_gompertz_has_no_finite_total() -> None:
    assert GompertzBaseline(rate=0.4, shape=0.3).total_hazard == np.inf


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ExponentialBaseline(rate=0.0),
        lambda: ExponentialBaseline(rate=-1.0),
        lambda: WeibullBaseline(shape=0.0, scale=1.0),
        lambda: WeibullBaseline(shape=1.0, scale=-2.0),
        lambda: GompertzBaseline(rate=-0.1, shape=0.2),
        lambda: GompertzBaseline(rate=0.4, shape=0.0),
        lambda: LogLogisticBaseline(shape=-1.0, scale=1.0),
        lambda: PiecewiseConstantBaseline([1.0], [0.5]),
        lambda: PiecewiseConstantBaseline([3.0, 1.0], [0.5, 1.0, 2.0]),
        lambda: PiecewiseConstantBaseline([1.0], [0.5, -1.0]),
    ],
)
def test_invalid_parameters_are_rejected(factory) -> None:
    with pytest.raises(ValidationError):
        factory()


def test_baselines_are_frozen_and_comparable() -> None:
    first = WeibullBaseline(shape=1.5, scale=2.0)
    second = WeibullBaseline(shape=1.5, scale=2.0)

    assert first == second
    with pytest.raises(Exception):
        first.shape = 2.0  # type: ignore[misc]


def test_registry_maps_names_to_classes() -> None:
    assert set(BASELINES) == {
        "exponential",
        "weibull",
        "gompertz",
        "log_logistic",
        "piecewise",
    }
    assert BASELINES["weibull"] is WeibullBaseline
