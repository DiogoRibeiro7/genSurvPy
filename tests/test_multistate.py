"""Tests for the general multistate engine.

The engine walks an arbitrary transition graph, so the things worth asserting
are the ones a specific model could not tell you: that each edge's intensity
comes back where it was set, that the competition between edges out of a state
has the right distribution, that the two clocks differ exactly where theory says
they should, and that the graph itself is validated.

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from gen_surv.baseline import ExponentialBaseline, GompertzBaseline, WeibullBaseline
from gen_surv.multistate import Transition, gen_multistate
from gen_surv.validation import ValidationError

_N = 40_000
_SEED = 20260824
_NO_CENSORING = 1e9

ILLNESS_DEATH = [
    Transition(1, 2, ExponentialBaseline(0.3), [0.0]),
    Transition(1, 3, ExponentialBaseline(0.2), [0.0]),
    Transition(2, 3, ExponentialBaseline(0.5), [0.0]),
]


def _intensity(frame, origin: int, destination: int) -> float:
    """Occurrence over exposure for one edge."""
    rows = frame[(frame["from_state"] == origin) & (frame["to_state"] == destination)]
    exposure = float((rows["stop"] - rows["start"]).sum())
    return int(rows["status"].sum()) / exposure


# --------------------------------------------------------------------------
# The process
# --------------------------------------------------------------------------


def test_every_edge_intensity_is_recovered() -> None:
    frame = gen_multistate(
        n=_N,
        transitions=ILLNESS_DEATH,
        cens_par=_NO_CENSORING,
        max_time=60.0,
        seed=_SEED,
    )

    for transition in ILLNESS_DEATH:
        declared = transition.baseline.rate  # type: ignore[attr-defined]
        estimate = _intensity(frame, transition.origin, transition.destination)
        np.testing.assert_allclose(estimate, declared, rtol=0.05)


def test_sojourn_in_a_state_is_exponential_in_the_summed_intensity() -> None:
    """Competing exponentials: the exit time has the summed rate."""
    frame = gen_multistate(
        n=_N,
        transitions=ILLNESS_DEATH,
        cens_par=_NO_CENSORING,
        max_time=60.0,
        seed=_SEED,
    )

    from_one = frame[(frame["from_state"] == 1) & (frame["to_state"] == 2)]
    sojourn = (from_one["stop"] - from_one["start"]).to_numpy()
    total = 0.3 + 0.2

    assert stats.kstest(sojourn * total, "expon").pvalue > 0.01


def test_destination_share_follows_the_competing_intensities() -> None:
    frame = gen_multistate(
        n=_N,
        transitions=ILLNESS_DEATH,
        cens_par=_NO_CENSORING,
        max_time=60.0,
        seed=_SEED,
    )

    from_one = frame[(frame["from_state"] == 1) & (frame["to_state"] == 2)]

    np.testing.assert_allclose(from_one["status"].mean(), 0.3 / 0.5, rtol=0.05)


def test_a_covariate_multiplies_one_edge_only() -> None:
    transitions = [
        Transition(1, 2, ExponentialBaseline(0.4), [np.log(2.0)]),
        Transition(1, 3, ExponentialBaseline(0.3), [0.0]),
    ]
    frame = gen_multistate(
        n=_N,
        transitions=transitions,
        covariate_dist="binary",
        covariate_params={"p": 0.5},
        cens_par=_NO_CENSORING,
        max_time=60.0,
        seed=_SEED,
    )

    rows = frame[(frame["from_state"] == 1) & (frame["to_state"] == 2)]
    rates = {}
    for label, mask in (("on", rows["X0"] > 0.5), ("off", rows["X0"] <= 0.5)):
        subset = rows[mask]
        rates[label] = int(subset["status"].sum()) / float(
            (subset["stop"] - subset["start"]).sum()
        )

    np.testing.assert_allclose(rates["on"] / rates["off"], 2.0, rtol=0.05)
    np.testing.assert_allclose(rates["off"], 0.4, rtol=0.05)


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------


def test_the_clock_is_irrelevant_for_a_constant_hazard() -> None:
    """A constant hazard is memoryless, so resetting it changes nothing."""
    common = dict(
        n=300,
        transitions=ILLNESS_DEATH,
        cens_par=_NO_CENSORING,
        max_time=30.0,
        seed=_SEED,
    )

    forward = gen_multistate(clock="forward", **common)
    reset = gen_multistate(clock="reset", **common)

    assert forward.equals(reset)


def test_the_clock_matters_for_a_rising_hazard() -> None:
    """On a forward clock the second sojourn starts partway up the curve."""
    transitions = [
        Transition(1, 2, WeibullBaseline(shape=2.0, scale=3.0), [0.0]),
        Transition(2, 3, WeibullBaseline(shape=2.0, scale=3.0), [0.0]),
    ]
    common = dict(
        n=8000,
        transitions=transitions,
        cens_par=_NO_CENSORING,
        max_time=30.0,
        seed=_SEED,
    )

    def mean_second_sojourn(clock: str) -> float:
        frame = gen_multistate(clock=clock, **common)
        rows = frame[(frame["from_state"] == 2) & (frame["status"] == 1)]
        return float((rows["stop"] - rows["start"]).mean())

    forward = mean_second_sojourn("forward")
    reset = mean_second_sojourn("reset")

    assert forward < reset / 1.5, (
        f"forward {forward:.3f} should be well below reset {reset:.3f}: on a "
        "forward clock the hazard has already climbed by the time state 2 is "
        "entered"
    )


# --------------------------------------------------------------------------
# Layouts and structure
# --------------------------------------------------------------------------


def test_interval_layout_is_the_counting_process_schema() -> None:
    frame = gen_multistate(n=200, transitions=ILLNESS_DEATH, seed=_SEED)

    assert list(frame.columns) == [
        "id",
        "start",
        "stop",
        "from_state",
        "to_state",
        "status",
        "X0",
    ]
    assert (frame["stop"] > frame["start"]).all()
    # One row per transition the subject was at risk of, so at most one fires
    # per occupancy.
    fired = frame.groupby(["id", "from_state"])["status"].sum()
    assert fired.max() <= 1


def test_panel_layout_opens_at_time_zero_in_the_initial_state() -> None:
    frame = gen_multistate(n=200, transitions=ILLNESS_DEATH, layout="panel", seed=_SEED)

    assert list(frame.columns) == ["id", "time", "state", "X0"]
    first = frame.sort_values(["id", "time"]).groupby("id").first()
    assert (first["time"] == 0.0).all()
    assert (first["state"] == 1).all()


def test_absorbing_states_end_follow_up() -> None:
    """Nothing is recorded after a state with no outgoing transition."""
    frame = gen_multistate(
        n=500,
        transitions=ILLNESS_DEATH,
        layout="panel",
        cens_par=_NO_CENSORING,
        max_time=100.0,
        seed=_SEED,
    )

    last = frame.sort_values("time").groupby("id").last()
    reached_three = last[last["state"] == 3]
    assert len(reached_three) > 0

    for subject in reached_three.index[:50]:
        states = frame[frame["id"] == subject].sort_values("time")["state"].tolist()
        assert states.count(3) == 1, "state 3 is absorbing and cannot repeat"
        assert states[-1] == 3


def test_follow_up_never_passes_the_administrative_end() -> None:
    frame = gen_multistate(
        n=500,
        transitions=ILLNESS_DEATH,
        cens_par=_NO_CENSORING,
        max_time=3.0,
        seed=_SEED,
    )

    assert frame["stop"].max() <= 3.0 + 1e-12


def test_a_cyclic_graph_is_allowed() -> None:
    """Recovery is a transition like any other; the engine does not assume a DAG."""
    transitions = [
        Transition(1, 2, ExponentialBaseline(0.8), [0.0]),
        Transition(2, 1, ExponentialBaseline(0.6), [0.0]),  # recovery
        Transition(2, 3, ExponentialBaseline(0.2), [0.0]),
    ]
    frame = gen_multistate(
        n=2000,
        transitions=transitions,
        cens_par=_NO_CENSORING,
        max_time=20.0,
        seed=_SEED,
    )

    revisits = frame[(frame["from_state"] == 1)].groupby("id").size()
    assert revisits.max() > 2, "expected some subjects to return to state 1"
    np.testing.assert_allclose(_intensity(frame, 2, 1), 0.6, rtol=0.05)


def test_a_gompertz_edge_can_stop_firing() -> None:
    """A declining Gompertz has a finite total hazard, so an edge can go quiet."""
    transitions = [
        Transition(1, 2, GompertzBaseline(rate=0.5, shape=-2.0), [0.0]),
        Transition(1, 3, ExponentialBaseline(0.05), [0.0]),
    ]
    frame = gen_multistate(
        n=2000,
        transitions=transitions,
        clock="reset",
        cens_par=_NO_CENSORING,
        max_time=50.0,
        seed=_SEED,
    )

    to_two = frame[(frame["from_state"] == 1) & (frame["to_state"] == 2)]
    assert 0 < to_two["status"].mean() < 1


# --------------------------------------------------------------------------
# Validation of the graph
# --------------------------------------------------------------------------


def test_a_transition_to_itself_is_rejected() -> None:
    with pytest.raises(ValidationError, match="differ from origin"):
        Transition(1, 1, ExponentialBaseline(0.5), [0.0])


@pytest.mark.parametrize("label", [1.5, "healthy", None, True])
def test_a_non_integer_state_label_is_rejected(label: object) -> None:
    """States are integers; a float or a name would compare unpredictably."""
    with pytest.raises(ValidationError, match="integer state label"):
        Transition(label, 2, ExponentialBaseline(0.5), [0.0])  # type: ignore[arg-type]


def test_a_runaway_cycle_is_reported_rather_than_hanging(monkeypatch) -> None:
    """A cycle with a high enough intensity must stop with an error.

    The cap exists so a graph that generates transitions faster than follow-up
    elapses cannot run forever. It is lowered here rather than building a
    graph that really would take ten thousand steps.
    """
    from gen_surv import multistate

    monkeypatch.setattr(multistate, "_MAX_TRANSITIONS", 3)
    transitions = [
        Transition(1, 2, ExponentialBaseline(50.0), [0.0]),
        Transition(2, 1, ExponentialBaseline(50.0), [0.0]),
    ]

    with pytest.raises(ValidationError, match="more than 3 transitions"):
        gen_multistate(
            n=50,
            transitions=transitions,
            cens_par=_NO_CENSORING,
            max_time=10.0,
            seed=_SEED,
        )


def test_a_non_baseline_is_rejected() -> None:
    with pytest.raises(ValidationError, match="BaselineHazard"):
        Transition(1, 2, object(), [0.0])  # type: ignore[arg-type]


def test_duplicate_edges_are_rejected() -> None:
    transitions = [
        Transition(1, 2, ExponentialBaseline(0.3), [0.0]),
        Transition(1, 2, ExponentialBaseline(0.4), [0.0]),
    ]
    with pytest.raises(ValidationError, match="twice"):
        gen_multistate(n=5, transitions=transitions, seed=_SEED)


def test_mismatched_coefficient_counts_are_rejected() -> None:
    transitions = [
        Transition(1, 2, ExponentialBaseline(0.3), [0.1]),
        Transition(1, 3, ExponentialBaseline(0.2), [0.1, 0.2]),
    ]
    with pytest.raises(ValidationError, match="same number of coefficients"):
        gen_multistate(n=5, transitions=transitions, seed=_SEED)


def test_an_initial_state_with_no_exit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="no outgoing transition"):
        gen_multistate(n=5, transitions=ILLNESS_DEATH, initial_state=3, seed=_SEED)


def test_an_empty_graph_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        gen_multistate(n=5, transitions=[], seed=_SEED)


@pytest.mark.parametrize("clock", ["sideways", ""])
def test_an_unknown_clock_is_rejected(clock: str) -> None:
    with pytest.raises(ValidationError, match="clock"):
        gen_multistate(n=5, transitions=ILLNESS_DEATH, clock=clock, seed=_SEED)  # type: ignore[arg-type]


def test_an_unknown_layout_is_rejected() -> None:
    with pytest.raises(ValidationError, match="layout"):
        gen_multistate(n=5, transitions=ILLNESS_DEATH, layout="wide", seed=_SEED)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_equal_seeds_give_equal_frames() -> None:
    common = dict(n=100, transitions=ILLNESS_DEATH)

    assert gen_multistate(seed=_SEED, **common).equals(
        gen_multistate(seed=_SEED, **common)
    )


def test_different_seeds_give_different_frames() -> None:
    common = dict(n=100, transitions=ILLNESS_DEATH)

    assert not gen_multistate(seed=_SEED, **common).equals(
        gen_multistate(seed=_SEED + 1, **common)
    )
