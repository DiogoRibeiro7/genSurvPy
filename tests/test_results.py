"""Tests for SimulationConfig, SimulationResult and simulate().

The interesting assertions are not that ``truth`` has keys, but that what it
reports is *true*: the latent times must reconstruct the observed ones, the
linear predictor must equal ``covariates @ betas``, and the recorded cure
status must match the column in the frame. A truth report that drifts from the
data it describes is worse than no truth report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gen_surv
from gen_surv import SimulationConfig, SimulationResult, generate, simulate
from gen_surv.validation import ValidationError

_SEED = 20260824


# --------------------------------------------------------------------------
# SimulationConfig
# --------------------------------------------------------------------------


def test_config_records_the_version_it_ran_under() -> None:
    config = SimulationConfig("cphm", {"n": 10, "seed": 1})

    assert config.version == gen_surv.__version__
    assert config.seed == 1


def test_config_seed_is_none_when_unseeded() -> None:
    assert SimulationConfig("cphm", {"n": 10}).seed is None


def test_config_replace_varies_one_thing() -> None:
    base = SimulationConfig("cphm", {"n": 100, "beta": 0.5})
    sweep = [base.replace(seed=s) for s in range(3)]

    assert [c.params["seed"] for c in sweep] == [0, 1, 2]
    assert all(c.params["n"] == 100 for c in sweep), "the rest must be carried over"
    assert "seed" not in base.params, "the original must not be mutated"


def test_config_round_trips_through_a_dict() -> None:
    config = SimulationConfig("cphm", {"n": 10, "beta": 0.5, "seed": 3})
    restored = SimulationConfig.from_dict(config.to_dict())

    assert restored == config


def test_config_run_reproduces_the_same_data() -> None:
    config = SimulationConfig(
        "cphm",
        {
            "n": 50,
            "beta": 0.5,
            "covariate_range": 2.0,
            "model_cens": "uniform",
            "cens_par": 1.0,
            "seed": _SEED,
        },
    )

    pd.testing.assert_frame_equal(config.run().data, config.run().data)


def test_config_rejects_an_empty_model() -> None:
    with pytest.raises(ValidationError):
        SimulationConfig("", {})


def test_config_is_frozen() -> None:
    config = SimulationConfig("cphm", {"n": 10})
    with pytest.raises(Exception):
        config.model = "aft_ln"  # type: ignore[misc]


# --------------------------------------------------------------------------
# SimulationResult
# --------------------------------------------------------------------------


def test_result_data_matches_the_plain_generator() -> None:
    """simulate() must not change what the generator produces."""
    kwargs = dict(
        n=100,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    )

    pd.testing.assert_frame_equal(
        simulate("cphm", **kwargs).data, generate(model="cphm", **kwargs)
    )


def test_simulate_returns_a_simulation_result() -> None:
    result = simulate(
        "cphm",
        n=10,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    )

    assert isinstance(result, SimulationResult)
    assert isinstance(result.config, SimulationConfig)
    assert isinstance(result.data, pd.DataFrame)


def test_result_counts_subjects_not_rows() -> None:
    result = simulate(
        "recurrent_events",
        n=50,
        baseline_params={"rate": 1.0},
        betas=[0.0, 0.0],
        followup_time=5.0,
        cens_par=1e9,
        seed=_SEED,
    )

    assert result.n_subjects == 50
    assert len(result) > 50, "a recurrent frame has more rows than subjects"


def test_truth_frame_keeps_only_per_subject_entries() -> None:
    result = simulate(
        "cphm",
        n=40,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    )
    frame = result.truth_frame()

    assert len(frame) == 40
    assert {"event_time", "censoring_time", "covariates"}.issubset(frame.columns)
    assert "beta" not in frame.columns, "a scalar is not a per-subject column"


def test_capturing_truth_does_not_leak_between_calls() -> None:
    first = simulate(
        "cphm",
        n=10,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    )
    second = simulate(
        "thmm",
        n=10,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.2, 0.3, 0.4],
        seed=_SEED,
    )

    assert "event_time" in first.truth
    assert "event_time" not in second.truth, "keys from an earlier call leaked"
    assert "transition_times" in second.truth


def test_plain_generator_calls_record_nothing() -> None:
    """Outside a capture block, recording must be inert."""
    from gen_surv._truth import _sink

    generate(
        model="cphm",
        n=10,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    )

    assert _sink.get() is None


# --------------------------------------------------------------------------
# Every model reports something, and what it reports is true
# --------------------------------------------------------------------------


CASES = {
    "cphm": dict(
        n=300,
        beta=0.5,
        covariate_range=2.0,
        model_cens="uniform",
        cens_par=1.0,
        seed=_SEED,
    ),
    "aft_ln": dict(
        n=300,
        beta=[0.5, -0.3],
        sigma=1.0,
        model_cens="uniform",
        cens_par=2.0,
        seed=_SEED,
    ),
    "aft_weibull": dict(
        n=300,
        beta=[0.5, -0.3],
        shape=1.4,
        scale=1.1,
        model_cens="uniform",
        cens_par=2.0,
        seed=_SEED,
    ),
    "aft_log_logistic": dict(
        n=300,
        beta=[0.5, -0.3],
        shape=1.3,
        scale=1.7,
        model_cens="uniform",
        cens_par=2.0,
        seed=_SEED,
    ),
    "piecewise_exponential": dict(
        n=300, breakpoints=[1.0], hazard_rates=[0.5, 1.5], seed=_SEED
    ),
    "competing_risks": dict(
        n=300,
        n_risks=2,
        baseline_hazards=[0.4, 0.2],
        betas=[[0.8, 0.0], [0.0, -0.5]],
        max_time=10.0,
        seed=_SEED,
    ),
    "competing_risks_weibull": dict(
        n=300,
        n_risks=2,
        shape_params=[1.2, 0.8],
        scale_params=[2.0, 1.5],
        max_time=10.0,
        seed=_SEED,
    ),
    "mixture_cure": dict(
        n=300,
        cure_fraction=0.3,
        baseline_hazard=0.8,
        betas_survival=[0.5, -0.2],
        betas_cure=[0.3, 0.1],
        seed=_SEED,
    ),
    "cmm": dict(
        n=200,
        model_cens="exponential",
        cens_par=2.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
        seed=_SEED,
    ),
    "thmm": dict(
        n=200,
        model_cens="exponential",
        cens_par=2.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.2, 0.3, 0.4],
        seed=_SEED,
    ),
    "tdcm": dict(
        n=200,
        dist="weibull",
        corr=0.5,
        dist_par=[1.0, 2.0, 1.0, 2.0],
        model_cens="uniform",
        cens_par=5.0,
        beta=[0.5, 0.3],
        lam=1.0,
        seed=_SEED,
    ),
    "recurrent_events": dict(
        n=200,
        baseline_params={"rate": 0.5},
        betas=[0.4, -0.2],
        followup_time=5.0,
        cens_par=8.0,
        seed=_SEED,
    ),
}


def test_every_registered_model_reports_truth() -> None:
    from gen_surv.interface import _model_map

    assert set(CASES) == set(_model_map), "a model was added without a truth case"

    for model, kwargs in CASES.items():
        result = simulate(model, **kwargs)
        assert result.truth, f"{model} reported no ground truth at all"
        assert result.config.model == model


@pytest.mark.parametrize(
    "model",
    ["cphm", "aft_ln", "aft_weibull", "aft_log_logistic", "piecewise_exponential"],
)
def test_latent_times_reconstruct_the_observed_ones(model: str) -> None:
    """``time`` is ``min(event, censoring)``, and ``status`` says which won."""
    result = simulate(model, **CASES[model])
    event = np.asarray(result.truth["event_time"])
    censoring = np.asarray(result.truth["censoring_time"])

    np.testing.assert_allclose(
        result.data["time"].to_numpy(), np.minimum(event, censoring)
    )
    np.testing.assert_array_equal(
        result.data["status"].to_numpy().astype(int), (event <= censoring).astype(int)
    )


@pytest.mark.parametrize(
    "model", ["aft_ln", "aft_weibull", "aft_log_logistic", "piecewise_exponential"]
)
def test_linear_predictor_is_covariates_times_betas(model: str) -> None:
    result = simulate(model, **CASES[model])

    np.testing.assert_allclose(
        np.asarray(result.truth["linear_predictor"]),
        np.asarray(result.truth["covariates"]) @ np.asarray(result.truth["betas"]),
    )


def test_cphm_linear_predictor_uses_its_scalar_beta() -> None:
    result = simulate("cphm", **CASES["cphm"])

    np.testing.assert_allclose(
        np.asarray(result.truth["linear_predictor"]),
        0.5 * result.data["X0"].to_numpy(),
    )


def test_randomly_drawn_betas_are_reported() -> None:
    """Without this there is no way to learn what coefficients were used.

    Several generators draw ``betas`` when the caller omits them, which
    otherwise makes the dataset useless for validating an estimator.
    """
    result = simulate(
        "piecewise_exponential",
        n=100,
        breakpoints=[1.0],
        hazard_rates=[0.5, 1.5],
        seed=_SEED,
    )

    betas = np.asarray(result.truth["betas"])
    assert betas.shape == (2,)
    np.testing.assert_allclose(
        np.asarray(result.truth["linear_predictor"]),
        np.asarray(result.truth["covariates"]) @ betas,
    )


@pytest.mark.parametrize("model", ["competing_risks", "competing_risks_weibull"])
def test_competing_risks_cause_times_explain_the_status(model: str) -> None:
    result = simulate(model, **CASES[model])
    cause_times = np.asarray(result.truth["cause_times"])
    status = result.data["status"].to_numpy()

    np.testing.assert_allclose(
        np.asarray(result.truth["first_event_time"]), cause_times.min(axis=1)
    )
    observed = status > 0
    np.testing.assert_array_equal(
        status[observed], (cause_times.argmin(axis=1) + 1)[observed]
    )


def test_mixture_cure_truth_matches_its_column() -> None:
    result = simulate("mixture_cure", **CASES["mixture_cure"])

    np.testing.assert_array_equal(
        np.asarray(result.truth["cured"]), result.data["cured"].to_numpy()
    )
    np.testing.assert_allclose(
        np.asarray(result.truth["cure_linear_predictor"]),
        np.asarray(result.truth["covariates"]) @ np.asarray(result.truth["betas_cure"]),
    )


@pytest.mark.parametrize("model", ["cmm", "thmm"])
def test_multistate_latent_times_explain_the_exit_from_state_one(model: str) -> None:
    result = simulate(model, **CASES[model])
    times = result.truth["transition_times"]
    censoring = np.asarray(result.truth["censoring_time"])

    expected = np.minimum(np.minimum(times["t12"], times["t13"]), censoring)

    if model == "thmm":
        observed = (
            result.data.sort_values(["id", "time"])
            .groupby("id")
            .nth(1)["time"]
            .to_numpy()
        )
    else:
        rows = result.data[
            (result.data["from_state"] == 1) & (result.data["to_state"] == 2)
        ].sort_values("id")
        observed = rows["stop"].to_numpy()

    np.testing.assert_allclose(observed, expected)


def test_tdcm_reports_the_crossover_time_the_frame_cannot() -> None:
    """The frame records only the covariate's value at exit, not when it switched."""
    result = simulate("tdcm", **CASES["tdcm"])

    crossover = np.asarray(result.truth["crossover_time"])
    assert crossover.shape == (200,)
    assert np.all(crossover > 0)
    np.testing.assert_allclose(
        np.asarray(result.truth["covariates"]), result.data["covariate"].to_numpy()
    )


def test_recurrent_followup_end_is_the_earlier_of_dropout_and_administrative_end() -> (
    None
):
    result = simulate("recurrent_events", **CASES["recurrent_events"])

    np.testing.assert_allclose(
        np.asarray(result.truth["followup_end"]),
        np.minimum(np.asarray(result.truth["dropout_time"]), 5.0),
    )


def test_recurrent_reports_the_baseline_object() -> None:
    from gen_surv.baseline import BaselineHazard

    result = simulate("recurrent_events", **CASES["recurrent_events"])

    assert isinstance(result.truth["baseline"], BaselineHazard)
