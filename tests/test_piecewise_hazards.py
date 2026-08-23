"""Distributional tests for the piecewise exponential generator.

These assert that each interval carries the hazard it was given, rather than
only that the returned frame has the right shape. The defect they guard against
shipped through 2.0.1: an event falling in a *middle* interval had its time
recomputed with the last hazard rate, because the trailing "no event yet" branch
ran even after the loop had broken out with a time already assigned.

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

import numpy as np
import pandas as pd

from gen_surv.piecewise import gen_piecewise_exponential

# Large enough to pin each interval's hazard, small enough to keep CI quick.
_N = 60_000
_SEED = 20260823

# Effectively no censoring: every event time is observed, so occurrence over
# exposure estimates the hazard directly.
_NO_CENSORING = 1e9


def _empirical_hazard(frame: pd.DataFrame, low: float, high: float) -> float:
    """Estimate the hazard on ``[low, high)`` as events over time at risk."""
    at_risk = frame.loc[frame["time"] > low, "time"]
    exposure = float(np.minimum(at_risk, high).sub(low).clip(lower=0).sum())
    events = int(
        ((frame["time"] > low) & (frame["time"] <= high) & (frame["status"] == 1)).sum()
    )
    return events / exposure


def _generate(breakpoints: list[float], hazard_rates: list[float]) -> pd.DataFrame:
    """Draw a sample whose hazard is the baseline alone, with no censoring."""
    return gen_piecewise_exponential(
        n=_N,
        breakpoints=breakpoints,
        hazard_rates=hazard_rates,
        betas=[0.0, 0.0],  # covariates present but inert, so hazard == baseline
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )


def test_single_breakpoint_hazards_match() -> None:
    """The two-interval case, which was already correct, stays correct."""
    breakpoints, rates = [1.0], [0.5, 2.0]
    frame = _generate(breakpoints, rates)

    edges = [0.0, *breakpoints, float(frame["time"].max())]
    for low, high, declared in zip(edges[:-1], edges[1:], rates):
        estimate = _empirical_hazard(frame, low, high)
        np.testing.assert_allclose(
            estimate,
            declared,
            rtol=0.05,
            err_msg=f"hazard on [{low}, {high})",
        )


def test_middle_interval_uses_its_own_hazard() -> None:
    """A middle interval must use its own rate, not the last one.

    This is the regression: with ``breakpoints=[1.0, 3.0]`` the interval
    ``[1, 3)`` came out at 0.2 -- the final rate -- instead of its declared 2.0,
    a tenfold error in the data with nothing in the frame to reveal it.
    """
    breakpoints, rates = [1.0, 3.0], [0.5, 2.0, 0.2]
    frame = _generate(breakpoints, rates)

    middle = _empirical_hazard(frame, 1.0, 3.0)

    np.testing.assert_allclose(middle, rates[1], rtol=0.05)
    assert abs(middle - rates[-1]) > 1.0, (
        f"middle interval hazard {middle:.3f} matches the last declared rate "
        f"{rates[-1]}, which is the pre-2.0.2 overwrite bug"
    )


def test_every_interval_matches_its_declared_hazard() -> None:
    """All three intervals of a two-breakpoint specification."""
    breakpoints, rates = [1.0, 3.0], [0.5, 2.0, 0.2]
    frame = _generate(breakpoints, rates)

    edges = [0.0, *breakpoints, float(frame["time"].max())]
    for low, high, declared in zip(edges[:-1], edges[1:], rates):
        estimate = _empirical_hazard(frame, low, high)
        # The open-ended final interval is populated by the few survivors of the
        # earlier ones, so it carries more Monte Carlo noise than the others.
        rtol = 0.2 if high == edges[-1] else 0.05
        np.testing.assert_allclose(
            estimate,
            declared,
            rtol=rtol,
            err_msg=f"hazard on [{low}, {high})",
        )


def test_many_breakpoints_stay_ordered() -> None:
    """With four intervals, a rising specification must produce rising hazards."""
    breakpoints, rates = [0.5, 1.0, 1.5], [0.2, 0.6, 1.8, 5.0]
    frame = _generate(breakpoints, rates)

    edges = [0.0, *breakpoints, float(frame["time"].max())]
    estimates = [
        _empirical_hazard(frame, low, high) for low, high in zip(edges[:-1], edges[1:])
    ]

    assert estimates == sorted(estimates), (
        f"hazards {[round(e, 3) for e in estimates]} are not increasing, though "
        f"the declared rates {rates} are"
    )
    for estimate, declared in zip(estimates[:-1], rates[:-1]):
        np.testing.assert_allclose(estimate, declared, rtol=0.1)


def test_covariates_scale_every_interval() -> None:
    """A positive coefficient must raise the hazard in all intervals alike."""
    breakpoints, rates = [1.0, 3.0], [0.5, 2.0, 0.2]

    baseline = _generate(breakpoints, rates)
    shifted = gen_piecewise_exponential(
        n=_N,
        breakpoints=breakpoints,
        hazard_rates=rates,
        betas=[0.5, 0.0],
        covariate_dist="binary",
        covariate_params={"p": 1.0},  # every subject has X0 == 1
        model_cens="uniform",
        cens_par=_NO_CENSORING,
        seed=_SEED,
    )

    factor = float(np.exp(0.5))
    for low, high in [(0.0, 1.0), (1.0, 3.0)]:
        base = _empirical_hazard(baseline, low, high)
        raised = _empirical_hazard(shifted, low, high)
        np.testing.assert_allclose(raised / base, factor, rtol=0.1)
