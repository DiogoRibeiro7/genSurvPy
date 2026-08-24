"""Tests for the recurrent event generator.

These check the process the generator claims to produce, not only the shape of
the frame: event counts against the theoretical mean of the counting process,
rate ratios against ``exp(beta)``, gap-time distributions against the stratum
effects, and the structural invariants of the counting-process layout.

All draws use fixed seeds, so a failure is a real regression rather than Monte
Carlo noise.
"""

import numpy as np
import pandas as pd
import pytest

from gen_surv import generate
from gen_surv.recurrent import gen_recurrent_events
from gen_surv.validation import ValidationError

# Large enough to pin a mean count to about half a percent.
_N = 20_000
_SEED = 20260823

# Effectively no random dropout, so follow-up ends at ``followup_time``.
_NO_DROPOUT = 1e9

_FOLLOWUP = 5.0


def _event_counts(frame: pd.DataFrame) -> pd.Series:
    """Number of observed events per subject."""
    return frame.groupby("id")["status"].sum()


# --------------------------------------------------------------------------
# Structure of the counting-process layout
# --------------------------------------------------------------------------


def test_columns_and_dtypes() -> None:
    frame = gen_recurrent_events(
        n=20, betas=[0.1, -0.1], followup_time=_FOLLOWUP, seed=_SEED
    )

    assert list(frame.columns) == ["id", "start", "stop", "status", "enum", "X0", "X1"]
    assert frame["id"].dtype == "int64"
    assert frame["status"].dtype == "int64"
    assert frame["enum"].dtype == "int64"


def test_intervals_are_contiguous_and_ordered() -> None:
    """Each subject's intervals must tile its follow-up without gaps."""
    frame = gen_recurrent_events(
        n=500,
        baseline_params={"rate": 1.0},
        betas=[0.1, 0.1],
        followup_time=6.0,
        cens_par=8.0,
        seed=_SEED,
    )

    for _, subject in frame.sort_values(["id", "start"]).groupby("id"):
        starts = subject["start"].to_numpy()
        stops = subject["stop"].to_numpy()

        assert starts[0] == 0.0, "follow-up must open at time zero"
        assert (stops > starts).all(), "every interval must have positive width"
        np.testing.assert_allclose(starts[1:], stops[:-1], err_msg="gap between rows")
        assert list(subject["enum"]) == list(range(1, len(subject) + 1))


def test_every_subject_appears_even_without_events() -> None:
    """A subject with no events still contributes its censored interval."""
    frame = gen_recurrent_events(
        n=300,
        baseline_params={"rate": 0.01},  # events are very unlikely
        betas=[0.0, 0.0],
        followup_time=1.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    assert frame["id"].nunique() == 300
    assert (_event_counts(frame) == 0).any(), "expected some event-free subjects"
    assert (frame.groupby("id").size() >= 1).all()


def test_follow_up_never_exceeds_the_administrative_end() -> None:
    frame = gen_recurrent_events(
        n=500,
        baseline_params={"rate": 2.0},
        betas=[0.0, 0.0],
        followup_time=3.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    assert frame["stop"].max() <= 3.0 + 1e-12
    np.testing.assert_allclose(frame.groupby("id")["stop"].max().to_numpy(), 3.0)


def test_last_row_of_each_subject_is_censored() -> None:
    frame = gen_recurrent_events(
        n=200,
        baseline_params={"rate": 1.0},
        betas=[0.0, 0.0],
        followup_time=4.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    last = frame.sort_values(["id", "start"]).groupby("id").last()
    assert (last["status"] == 0).all()


# --------------------------------------------------------------------------
# Andersen-Gill: the counting process itself
# --------------------------------------------------------------------------


def test_exponential_baseline_gives_a_poisson_count() -> None:
    """With a constant intensity, N(T) is Poisson with mean ``rate * T``.

    Poisson means the variance equals the mean, which distinguishes this from a
    process that merely gets the average right.
    """
    rate = 0.5
    frame = gen_recurrent_events(
        n=_N,
        baseline_params={"rate": rate},
        betas=[0.0, 0.0],
        followup_time=_FOLLOWUP,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    counts = _event_counts(frame)
    expected = rate * _FOLLOWUP

    np.testing.assert_allclose(counts.mean(), expected, rtol=0.05)
    np.testing.assert_allclose(counts.var(ddof=1), expected, rtol=0.10)


def test_weibull_baseline_count_matches_its_cumulative_hazard() -> None:
    """E[N(T)] is the integrated baseline hazard, here ``(T / scale) ** shape``."""
    shape, scale = 1.5, 2.0
    frame = gen_recurrent_events(
        n=_N,
        baseline="weibull",
        baseline_params={"shape": shape, "scale": scale},
        betas=[0.0, 0.0],
        followup_time=_FOLLOWUP,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    expected = (_FOLLOWUP / scale) ** shape
    np.testing.assert_allclose(_event_counts(frame).mean(), expected, rtol=0.05)


def test_gompertz_baseline_count_matches_its_cumulative_hazard() -> None:
    """E[N(T)] = (rate / shape) * (exp(shape * T) - 1)."""
    rate, shape = 0.3, 0.2
    frame = gen_recurrent_events(
        n=_N,
        baseline="gompertz",
        baseline_params={"rate": rate, "shape": shape},
        betas=[0.0, 0.0],
        followup_time=_FOLLOWUP,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    expected = rate / shape * (np.exp(shape * _FOLLOWUP) - 1.0)
    np.testing.assert_allclose(_event_counts(frame).mean(), expected, rtol=0.05)


def test_covariate_multiplies_the_intensity_by_exp_beta() -> None:
    """A binary covariate with ``beta = log 2`` must double the event rate."""
    frame = gen_recurrent_events(
        n=_N,
        baseline_params={"rate": 0.5},
        betas=[np.log(2.0), 0.0],
        covariate_dist="binary",
        covariate_params={"p": 0.5},
        followup_time=_FOLLOWUP,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    exposed = frame["X0"] > 0.5
    rates = {
        group: frame.loc[mask, "status"].sum()
        / (frame.loc[mask, "stop"] - frame.loc[mask, "start"]).sum()
        for group, mask in (("treated", exposed), ("control", ~exposed))
    }

    np.testing.assert_allclose(rates["treated"] / rates["control"], 2.0, rtol=0.05)


# --------------------------------------------------------------------------
# Prentice-Williams-Peterson: stratum effects and the clock
# --------------------------------------------------------------------------


def test_stratum_effects_scale_the_gap_times() -> None:
    """Doubling the intensity after the first event halves the mean gap."""
    frame = gen_recurrent_events(
        n=_N,
        process="pwp_gt",
        baseline_params={"rate": 1.0},
        betas=[0.0, 0.0],
        stratum_effects=[1.0, 2.0],
        followup_time=20.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    events = frame[frame["status"] == 1].assign(gap=lambda d: d["stop"] - d["start"])

    np.testing.assert_allclose(
        events.loc[events["enum"] == 1, "gap"].mean(), 1.0, rtol=0.05
    )
    np.testing.assert_allclose(
        events.loc[events["enum"] > 1, "gap"].mean(), 0.5, rtol=0.05
    )


def test_unit_stratum_effects_reduce_pwp_to_andersen_gill() -> None:
    """PWP in total time with no stratum effect is Andersen-Gill."""
    common = dict(
        n=200,
        baseline_params={"rate": 0.7},
        betas=[0.3, -0.1],
        followup_time=4.0,
        seed=_SEED,
    )

    ag = gen_recurrent_events(process="ag", **common)
    pwp = gen_recurrent_events(process="pwp_tt", stratum_effects=[1.0], **common)

    pd.testing.assert_frame_equal(ag, pwp)


def test_exponential_baseline_makes_the_clock_irrelevant() -> None:
    """A constant hazard is memoryless, so resetting the clock changes nothing."""
    common = dict(
        n=300,
        baseline_params={"rate": 0.8},
        betas=[0.2, -0.1],
        followup_time=6.0,
        seed=_SEED,
    )

    forward = gen_recurrent_events(process="ag", **common)
    reset = gen_recurrent_events(process="pwp_gt", **common)

    pd.testing.assert_frame_equal(forward, reset, atol=1e-9)


def test_resetting_the_clock_matters_for_a_rising_hazard() -> None:
    """With a Weibull shape above 1, the forward clock produces far more events.

    On a forward clock the hazard keeps climbing with time since entry; on a
    reset clock every event returns it to the start of the curve.
    """
    common = dict(
        n=2000,
        baseline="weibull",
        baseline_params={"shape": 2.0, "scale": 2.0},
        betas=[0.0, 0.0],
        followup_time=6.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    forward = _event_counts(gen_recurrent_events(process="ag", **common)).mean()
    reset = _event_counts(gen_recurrent_events(process="pwp_gt", **common)).mean()

    assert forward > 2 * reset, (
        f"forward clock produced {forward:.2f} events per subject against "
        f"{reset:.2f} on a reset clock; the two clocks should differ sharply "
        "for a rising hazard"
    )


def test_stratum_effects_are_rejected_by_andersen_gill() -> None:
    """``ag`` has no event-number dependence, so per-event effects must raise.

    Applying them quietly would produce PWP data under an Andersen-Gill label;
    dropping them quietly would discard an argument the caller meant.
    """
    with pytest.raises(ValidationError, match="not applicable to process='ag'"):
        gen_recurrent_events(
            n=200,
            process="ag",
            baseline_params={"rate": 1.0},
            betas=[0.0, 0.0],
            stratum_effects=[1.0, 5.0],
            followup_time=5.0,
            seed=_SEED,
        )


# --------------------------------------------------------------------------
# Caps, censoring and reproducibility
# --------------------------------------------------------------------------


def test_max_events_caps_the_process() -> None:
    frame = gen_recurrent_events(
        n=500,
        baseline_params={"rate": 2.0},
        betas=[0.0, 0.0],
        followup_time=10.0,
        cens_par=_NO_DROPOUT,
        max_events=3,
        seed=_SEED,
    )

    counts = _event_counts(frame)
    assert counts.max() == 3
    assert (counts == 3).mean() > 0.5, "the cap should bind for most subjects"


def test_random_dropout_shortens_follow_up() -> None:
    """Dropout applies on top of the administrative end."""
    heavy = gen_recurrent_events(
        n=2000,
        baseline_params={"rate": 1.0},
        betas=[0.0, 0.0],
        followup_time=10.0,
        model_cens="uniform",
        cens_par=2.0,
        seed=_SEED,
    )
    light = gen_recurrent_events(
        n=2000,
        baseline_params={"rate": 1.0},
        betas=[0.0, 0.0],
        followup_time=10.0,
        model_cens="uniform",
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    assert heavy["stop"].max() <= 2.0 + 1e-12
    assert _event_counts(heavy).mean() < _event_counts(light).mean()


def test_equal_seeds_give_equal_frames() -> None:
    common = dict(n=50, betas=[0.3, -0.2], followup_time=4.0)

    pd.testing.assert_frame_equal(
        gen_recurrent_events(seed=_SEED, **common),
        gen_recurrent_events(seed=_SEED, **common),
    )


def test_different_seeds_give_different_frames() -> None:
    common = dict(n=50, betas=[0.3, -0.2], followup_time=4.0)

    first = gen_recurrent_events(seed=_SEED, **common)
    second = gen_recurrent_events(seed=_SEED + 1, **common)

    assert not first.equals(second)


def test_generator_is_registered_with_the_dispatcher() -> None:
    frame = generate(
        model="recurrent_events",
        n=10,
        betas=[0.2, 0.1],
        followup_time=3.0,
        seed=_SEED,
    )

    assert {"id", "start", "stop", "status", "enum"}.issubset(frame.columns)


# --------------------------------------------------------------------------
# Baseline hazards supplied as objects
# --------------------------------------------------------------------------


def test_accepts_a_baseline_the_generator_does_not_name() -> None:
    """Any object implementing the protocol works, not only the three names.

    The mean event count is the integrated baseline hazard, so this checks the
    supplied object is actually driving the sampler rather than being ignored.
    """
    from gen_surv.baseline import LogLogisticBaseline

    baseline = LogLogisticBaseline(shape=2.0, scale=1.5)
    frame = gen_recurrent_events(
        n=_N,
        baseline=baseline,
        betas=[0.0, 0.0],
        followup_time=6.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    expected = float(baseline.cumulative_hazard(6.0))
    np.testing.assert_allclose(_event_counts(frame).mean(), expected, rtol=0.05)


def test_accepts_a_piecewise_baseline_object() -> None:
    from gen_surv.baseline import PiecewiseConstantBaseline

    baseline = PiecewiseConstantBaseline([1.0, 3.0], [0.5, 2.0, 0.2])
    frame = gen_recurrent_events(
        n=_N,
        baseline=baseline,
        betas=[0.0, 0.0],
        followup_time=5.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    np.testing.assert_allclose(
        _event_counts(frame).mean(), float(baseline.cumulative_hazard(5.0)), rtol=0.05
    )


def test_named_baseline_and_object_agree() -> None:
    """The string form is a shortcut for constructing the object."""
    from gen_surv.baseline import WeibullBaseline

    common = dict(
        n=200, betas=[0.2, -0.1], followup_time=5.0, cens_par=_NO_DROPOUT, seed=_SEED
    )

    by_name = gen_recurrent_events(
        baseline="weibull", baseline_params={"shape": 1.3, "scale": 2.0}, **common
    )
    by_object = gen_recurrent_events(
        baseline=WeibullBaseline(shape=1.3, scale=2.0), **common
    )

    pd.testing.assert_frame_equal(by_name, by_object)


def test_baseline_object_cannot_be_combined_with_parameters() -> None:
    from gen_surv.baseline import ExponentialBaseline

    with pytest.raises(ValidationError, match="cannot be combined"):
        gen_recurrent_events(
            n=10,
            baseline=ExponentialBaseline(rate=1.0),
            baseline_params={"rate": 2.0},
            betas=[0.0, 0.0],
            followup_time=2.0,
            seed=_SEED,
        )


def test_an_object_that_is_not_a_baseline_is_rejected() -> None:
    with pytest.raises(ValidationError, match="implementing"):
        gen_recurrent_events(
            n=10, baseline=object(), betas=[0.0, 0.0], followup_time=2.0, seed=_SEED
        )


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"process": "cox"},
        {"baseline": "lognormal"},
        {"followup_time": 0.0},
        {"cens_par": -1.0},
        {"model_cens": "gamma"},
        {"max_events": 0},
        {"n_covariates": 0},
        {"stratum_effects": []},
        {"stratum_effects": [1.0, -2.0]},
        {"baseline_params": {"rate": 0.0}},
        {"baseline_params": {"lambda": 1.0}},
        {"baseline": "gompertz", "baseline_params": {"rate": 1.0, "shape": 0.0}},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict) -> None:
    defaults = dict(n=10, betas=[0.1, 0.1], followup_time=2.0, seed=_SEED)
    defaults.update(kwargs)

    with pytest.raises(ValidationError):
        gen_recurrent_events(**defaults)


def test_gompertz_accepts_a_negative_shape() -> None:
    """A declining Gompertz hazard is legitimate, and has a finite total hazard."""
    frame = gen_recurrent_events(
        n=1000,
        baseline="gompertz",
        baseline_params={"rate": 1.0, "shape": -0.5},
        betas=[0.0, 0.0],
        followup_time=100.0,
        cens_par=_NO_DROPOUT,
        seed=_SEED,
    )

    # Total hazard converges to rate / |shape| = 2, so counts stay small even
    # over very long follow-up.
    assert _event_counts(frame).mean() < 4.0


def test_betas_length_sets_the_covariate_count() -> None:
    frame = gen_recurrent_events(
        n=20, betas=[0.1, 0.2, -0.3], followup_time=2.0, seed=_SEED
    )

    assert {"X0", "X1", "X2"}.issubset(frame.columns)
    assert "X3" not in frame.columns
