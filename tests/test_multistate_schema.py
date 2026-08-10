"""Structural tests for the multistate generators.

`gen_cmm` emits counting-process intervals and `gen_thmm` emits a state-per-
observation panel. The two layouts differ deliberately, matching `genCMM` and
`genTHMM` in the R package respectively, so they are checked separately.

Releases up to 1.3.0 collapsed both to one row per subject and reported only the
first transition, which left several declared parameters with no effect on the
output. The parameter-influence tests below are the direct guards for that.
"""

import numpy as np
import pandas as pd
import pytest

from gen_surv.cmm import gen_cmm
from gen_surv.thmm import gen_thmm

_CMM = {
    "n": 200,
    "model_cens": "uniform",
    "cens_par": 3.0,
    "beta": [0.3, -0.2, 0.1],
    "covariate_range": 1.0,
    "rate": [0.6, 1.0, 0.4, 1.0, 0.8, 1.0],
    "seed": 20260810,
}

_THMM = {
    "n": 200,
    "model_cens": "uniform",
    "cens_par": 5.0,
    "beta": [0.1, 0.2, 0.3],
    "covariate_range": 1.0,
    "rate": [0.6, 0.3, 0.5],
    "seed": 20260810,
}


# --------------------------------------------------------------------------
# CMM: counting-process intervals
# --------------------------------------------------------------------------


def test_cmm_subjects_contribute_two_or_three_rows() -> None:
    """One row per at-risk transition, never a single collapsed row."""
    counts = gen_cmm(**_CMM).groupby("id").size()
    assert set(counts.unique()) <= {2, 3}
    assert (counts >= 2).all()


def test_cmm_both_competing_transitions_are_always_at_risk() -> None:
    """While in state 1 a subject risks 1->2 and 1->3 over the same interval."""
    frame = gen_cmm(**_CMM)

    for _, subject in frame.groupby("id"):
        from_1 = subject[subject["from_state"] == 1]
        assert len(from_1) == 2
        assert set(from_1["to_state"]) == {2, 3}
        # Both rows end when the subject leaves state 1.
        assert from_1["stop"].nunique() == 1
        assert (from_1["start"] == 0.0).all()
        # At most one of the two competing transitions can be the one observed.
        assert from_1["status"].sum() <= 1


def test_cmm_state_2_row_exists_exactly_when_illness_was_observed() -> None:
    """A 2->3 row appears if and only if the 1->2 transition was an event."""
    frame = gen_cmm(**_CMM)

    for _, subject in frame.groupby("id"):
        illness = subject[(subject["from_state"] == 1) & (subject["to_state"] == 2)]
        reached_2 = bool(illness["status"].iloc[0])
        assert (subject["from_state"] == 2).any() == reached_2


def test_cmm_clock_resets_on_entry_to_state_2() -> None:
    """The 2->3 interval starts when the subject left state 1."""
    frame = gen_cmm(**_CMM)

    for _, subject in frame.groupby("id"):
        from_2 = subject[subject["from_state"] == 2]
        if from_2.empty:
            continue
        exit_1 = subject[subject["from_state"] == 1]["stop"].iloc[0]
        assert from_2["start"].iloc[0] == pytest.approx(exit_1)


def test_cmm_intervals_are_positive_and_ordered() -> None:
    """Every interval has positive length and rows are sorted."""
    frame = gen_cmm(**_CMM)

    assert (frame["stop"] > frame["start"]).all()
    assert frame.equals(
        frame.sort_values(["id", "start", "to_state"], kind="stable").reset_index(
            drop=True
        )
    )


def test_cmm_covariate_is_constant_within_a_subject() -> None:
    """A subject's covariate must not change between its rows."""
    assert (gen_cmm(**_CMM).groupby("id")["X0"].nunique() == 1).all()


@pytest.mark.parametrize("index", [4, 5])
def test_cmm_uses_the_state_2_rate_parameters(index: int) -> None:
    """rate[4] and rate[5] must reach the output.

    Up to 1.3.0 the 2->3 draw was discarded, so these two had no effect at all.
    """
    baseline = gen_cmm(**_CMM)

    changed = dict(_CMM)
    rate = list(_CMM["rate"])
    rate[index] = rate[index] * 4.0
    changed["rate"] = rate

    assert not gen_cmm(**changed).equals(baseline)


def test_cmm_uses_the_third_coefficient() -> None:
    """beta[2] governs the 2->3 intensity and must reach the output."""
    baseline = gen_cmm(**_CMM)

    changed = dict(_CMM)
    changed["beta"] = [0.3, -0.2, 2.5]

    assert not gen_cmm(**changed).equals(baseline)


# --------------------------------------------------------------------------
# THMM: state-per-observation panel
# --------------------------------------------------------------------------


def test_thmm_subjects_contribute_two_or_three_rows() -> None:
    """The full trajectory is emitted, not just the final observation."""
    counts = gen_thmm(**_THMM).groupby("id").size()
    assert set(counts.unique()) <= {2, 3}
    assert (counts >= 2).all()


def test_thmm_every_trajectory_starts_healthy_at_time_zero() -> None:
    """Entry into state 1 at time 0 is always recorded."""
    for _, subject in gen_thmm(**_THMM).groupby("id"):
        assert subject["time"].iloc[0] == 0.0
        assert subject["state"].iloc[0] == 1


def test_thmm_trajectories_are_monotone() -> None:
    """Time advances and the state never moves backwards."""
    for _, subject in gen_thmm(**_THMM).groupby("id"):
        assert subject["time"].is_monotonic_increasing
        assert subject["state"].is_monotonic_increasing


def test_thmm_death_is_terminal() -> None:
    """State 3 can only ever be the final observation."""
    for _, subject in gen_thmm(**_THMM).groupby("id"):
        states = list(subject["state"])
        if 3 in states:
            assert states.index(3) == len(states) - 1


def test_thmm_three_row_trajectories_are_those_passing_through_illness() -> None:
    """A third observation exists exactly when the subject reached state 2."""
    for _, subject in gen_thmm(**_THMM).groupby("id"):
        assert (len(subject) == 3) == (2 in set(subject["state"]))


def test_thmm_states_are_within_the_model() -> None:
    """Only the three modelled states may appear."""
    assert set(gen_thmm(**_THMM)["state"].unique()) <= {1, 2, 3}


def test_thmm_uses_the_state_2_rate() -> None:
    """rate[2] governs the 2->3 intensity and must reach the output.

    Up to 1.3.0 the 2->3 draw was discarded, so this had no effect at all.
    """
    baseline = gen_thmm(**_THMM)

    changed = dict(_THMM)
    changed["rate"] = [0.6, 0.3, 4.0]

    assert not baseline.equals(gen_thmm(**changed))


def test_thmm_uses_the_third_coefficient() -> None:
    """beta[2] governs the 2->3 intensity and must reach the output."""
    baseline = gen_thmm(**_THMM)

    changed = dict(_THMM)
    changed["beta"] = [0.1, 0.2, 2.5]

    assert not baseline.equals(gen_thmm(**changed))


# --------------------------------------------------------------------------
# The two layouts are deliberately different
# --------------------------------------------------------------------------


def test_the_two_generators_use_their_documented_layouts() -> None:
    """CMM emits intervals; THMM emits a state panel. This is intentional."""
    assert list(gen_cmm(**_CMM).columns) == [
        "id",
        "start",
        "stop",
        "from_state",
        "to_state",
        "status",
        "X0",
    ]
    assert list(gen_thmm(**_THMM).columns) == ["id", "time", "state", "X0"]


def test_both_generators_are_reproducible() -> None:
    """A seed reproduces the whole frame for both layouts."""
    pd.testing.assert_frame_equal(gen_cmm(**_CMM), gen_cmm(**_CMM))
    pd.testing.assert_frame_equal(gen_thmm(**_THMM), gen_thmm(**_THMM))


def test_both_generators_produce_some_of_every_transition() -> None:
    """With these parameters all three transitions should be observed.

    A weak check, but it catches a generator that silently never leaves state 1.
    """
    cmm = gen_cmm(**_CMM)
    observed = cmm[cmm["status"] == 1]
    assert {(1, 2), (1, 3), (2, 3)} <= set(
        zip(observed["from_state"], observed["to_state"])
    )

    assert {1, 2, 3} <= set(gen_thmm(**_THMM)["state"].unique())


def test_row_count_scales_with_subjects_and_ids_stay_in_range() -> None:
    """More subjects means more rows, and ids remain 0..n-1."""
    small = gen_cmm(**{**_CMM, "n": 10})
    large = gen_cmm(**{**_CMM, "n": 40})
    assert small["id"].nunique() == 10
    assert large["id"].nunique() == 40
    assert len(large) > len(small)
    assert np.isin(small["id"].unique(), np.arange(10)).all()
