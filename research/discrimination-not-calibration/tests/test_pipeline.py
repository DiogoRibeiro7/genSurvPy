"""Seeds, censoring calibration, configuration, metrics and aggregation.

The metric tests use hand-computable cases rather than fixtures. If a predicted
curve sits a constant ``c`` above the truth across ``[0, tau]`` then

.. math::

    ISE = \\int_0^\\tau c^2\\,dt = c^2\\tau,
    \\qquad
    IAE = |c|\\,\\tau,

which can be checked exactly. A fixture would only confirm that the code still
does what it did when the fixture was written, including if that was wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from survival_misspec.aggregation import (
    aggregate,
    completed_cells,
    failure_rates,
    mcse,
    replications_for_precision,
    write_raw,
)
from survival_misspec.config import (
    EstimatorConfig,
    MetricsConfig,
    ScenarioConfig,
    StudyConfig,
    content_hash,
)
from survival_misspec.estimators import ADAPTERS, build_estimator, fit_estimator
from survival_misspec.metrics import (
    integrated_absolute_error,
    integrated_squared_error,
)
from survival_misspec.simulation import calibrate_censoring, derive_seed, draw_replicate

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------


def test_seeds_depend_only_on_identifiers() -> None:
    assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)
    assert derive_seed(42, "a", 1) != derive_seed(42, "a", 2)
    assert derive_seed(42, "a", 1) != derive_seed(42, "b", 1)
    assert derive_seed(42, "a", 1) != derive_seed(43, "a", 1)


def test_train_and_evaluation_streams_differ() -> None:
    """Fitting and scoring on the same draw would confound overfitting."""
    assert derive_seed(42, "a", 1, "train") != derive_seed(42, "a", 1, "eval")


def test_seeds_do_not_depend_on_the_order_scenarios_are_declared() -> None:
    """The property that makes parallel and resumed runs identical.

    A counter-based scheme would give replicate 3 of scenario B a different
    seed depending on how many scenarios preceded it, so inserting a scenario
    would silently change data already collected for others.
    """
    first = [derive_seed(7, name, 0) for name in ("alpha", "beta", "gamma")]
    second = [derive_seed(7, name, 0) for name in ("gamma", "alpha", "beta")]
    assert dict(zip(("alpha", "beta", "gamma"), first)) == {
        "gamma": second[0],
        "alpha": second[1],
        "beta": second[2],
    }


def test_the_same_cell_reproduces_the_same_data() -> None:
    params = {
        "beta": 0.5,
        "covariate_range": 2.0,
        "model_cens": "uniform",
        "cens_par": 2.0,
    }
    first = draw_replicate("cphm", params, 100, "s", 3, 99)
    second = draw_replicate("cphm", params, 100, "s", 3, 99)
    pd.testing.assert_frame_equal(first.data, second.data)


# ---------------------------------------------------------------------------
# Censoring calibration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.1, 0.3, 0.5, 0.7])
def test_calibration_hits_the_target_censoring_rate(target: float) -> None:
    params = {"beta": 0.5, "covariate_range": 2.0, "model_cens": "uniform"}
    calibration = calibrate_censoring("cphm", params, target, n=4000)

    assert calibration.feasible
    assert (
        calibration.error <= 0.02
    ), f"asked for {target:.0%} censoring, achieved {calibration.achieved:.1%}"


def test_calibration_reports_an_infeasible_target_instead_of_approximating() -> None:
    """A cured subject never fails, so censoring has a floor.

    Returning the nearest achievable value would put a scenario in the design
    whose censoring level is not the one it claims, and every comparison across
    the censoring factor would then be against a mislabelled cell.
    """
    params = {
        "cure_fraction": 0.3,
        "baseline_hazard": 0.7,
        "betas_survival": [0.5, -0.3],
        "betas_cure": [0.4, 0.2],
        "model_cens": "uniform",
    }
    calibration = calibrate_censoring("mixture_cure", params, 0.05, n=3000)

    assert not calibration.feasible
    assert calibration.achieved > 0.25
    assert "without an event" in calibration.reason


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_content_hash_ignores_key_order() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


def test_content_hash_changes_with_content() -> None:
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_scenario_rejects_an_unsupported_dgp() -> None:
    with pytest.raises(ValueError, match="tdcm"):
        ScenarioConfig(
            scenario_id="x",
            dgp="tdcm",
            n=100,
            target_censoring=0.3,
            effect_size=0.5,
            params={},
        )


def test_scenario_rejects_impossible_censoring() -> None:
    with pytest.raises(ValueError, match="target_censoring"):
        ScenarioConfig(
            scenario_id="x",
            dgp="cphm",
            n=100,
            target_censoring=1.0,
            effect_size=0.5,
            params={},
        )


def _study(scenarios, estimators) -> StudyConfig:
    return StudyConfig(
        paper_id="p",
        master_seed=1,
        n_replications=2,
        scenarios=tuple(scenarios),
        estimators=tuple(estimators),
        metrics=MetricsConfig(0.8, 11, (0.5,), ("mise",)),
    )


def test_duplicate_scenario_ids_are_rejected() -> None:
    scenario = ScenarioConfig("dup", "cphm", 100, 0.3, 0.5, {})
    estimator = EstimatorConfig("cox", "cox_ph")
    with pytest.raises(ValueError, match="duplicate scenario_id"):
        _study([scenario, scenario], [estimator])


def test_study_hash_is_stable_and_sensitive() -> None:
    a = ScenarioConfig("a", "cphm", 100, 0.3, 0.5, {})
    b = ScenarioConfig("b", "cphm", 200, 0.3, 0.5, {})
    estimator = EstimatorConfig("cox", "cox_ph")

    assert _study([a, b], [estimator]).hash == _study([b, a], [estimator]).hash
    assert _study([a], [estimator]).hash != _study([a, b], [estimator]).hash


# ---------------------------------------------------------------------------
# Metrics, against hand-computable cases
# ---------------------------------------------------------------------------


def test_integrated_errors_match_the_closed_form_for_a_constant_offset() -> None:
    tau = 2.0
    grid = np.linspace(0.0, tau, 201)
    truth = np.tile(np.linspace(1.0, 0.4, grid.size), (3, 1))
    offset = np.array([[0.1], [-0.05], [0.0]])
    predicted = truth + offset

    ise = integrated_squared_error(predicted, truth, grid, tau)
    iae = integrated_absolute_error(predicted, truth, grid, tau)

    np.testing.assert_allclose(ise, (offset.ravel() ** 2) * tau, rtol=1e-10)
    np.testing.assert_allclose(iae, np.abs(offset.ravel()) * tau, rtol=1e-10)


def test_integrated_error_is_zero_for_a_perfect_prediction() -> None:
    grid = np.linspace(0.0, 1.0, 51)
    truth = np.tile(np.exp(-grid), (4, 1))
    assert integrated_squared_error(truth, truth, grid, 1.0).max() == 0.0


def test_integration_respects_tau_and_ignores_later_error() -> None:
    """Error beyond the horizon must not enter the loss.

    tau is prespecified; if the integral silently ran past it, the reported
    loss would depend on how far the grid happened to extend.
    """
    grid = np.linspace(0.0, 4.0, 401)
    truth = np.zeros((1, grid.size))
    predicted = np.where(grid > 2.0, 1.0, 0.0).reshape(1, -1)

    assert integrated_squared_error(predicted, truth, grid, 2.0)[0] == pytest.approx(
        0.0, abs=1e-12
    )
    assert integrated_squared_error(predicted, truth, grid, 4.0)[0] > 1.9


def test_a_horizon_below_the_grid_is_an_error_not_a_silent_zero() -> None:
    grid = np.linspace(0.0, 1.0, 11)
    truth = np.zeros((1, grid.size))
    with pytest.raises(ValueError, match="fewer than two points"):
        integrated_squared_error(truth, truth, grid, -1.0)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("adapter", sorted(ADAPTERS))
def test_every_adapter_predicts_a_valid_survival_surface(adapter: str) -> None:
    params = {
        "beta": 0.6,
        "covariate_range": 2.0,
        "model_cens": "uniform",
        "cens_par": 2.0,
    }
    replicate = draw_replicate("cphm", params, 300, "s", 0, 5)
    grid = np.linspace(0.0, 1.0, 21)

    options = {"n_estimators": 20} if "forest" in adapter or "boost" in adapter else {}
    fitted = fit_estimator(
        adapter,
        adapter,
        options,
        replicate.covariates,
        replicate.observed_time,
        replicate.event,
    )
    assert fitted.fitted, fitted.error

    surface = fitted.model.predict_survival(replicate.covariates, grid)
    assert surface.shape == (300, grid.size)
    assert np.all(surface >= 0.0) and np.all(surface <= 1.0)
    assert np.all(np.diff(surface, axis=1) <= 1e-12), "predicted survival increases"

    risk = fitted.model.predict_risk(replicate.covariates)
    assert risk.shape == (300,) and np.all(np.isfinite(risk))


def test_a_failed_fit_is_recorded_rather_than_raised() -> None:
    """Failure is data about the estimator, not an exception to escape with."""
    fitted = fit_estimator(
        "cox", "cox_ph", {}, np.zeros((5, 1)), np.array([1.0] * 5), np.ones(5, bool)
    )
    assert fitted.failed
    assert fitted.error_type
    assert fitted.runtime_seconds >= 0.0


def test_an_unknown_adapter_is_rejected_by_name() -> None:
    with pytest.raises(KeyError, match="not_a_model"):
        build_estimator("not_a_model")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_mcse_matches_the_definition() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert mcse(values) == pytest.approx(np.std(values, ddof=1) / 2.0)


def test_mcse_needs_more_than_one_value() -> None:
    assert np.isnan(mcse([1.0]))


def test_replications_for_precision_inverts_the_mcse_formula() -> None:
    values = np.random.default_rng(0).normal(size=500)
    required = replications_for_precision(values, target_mcse=0.01)
    assert required == pytest.approx((np.std(values, ddof=1) / 0.01) ** 2, rel=0.01)


def _raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 0,
                "fitted": True,
                "scored": True,
                "mise": 0.10,
                "dgp": "cphm",
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 1,
                "fitted": True,
                "scored": True,
                "mise": 0.20,
                "dgp": "cphm",
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 2,
                "fitted": False,
                "scored": False,
                "mise": np.nan,
                "dgp": "cphm",
            },
        ]
    )


def test_aggregate_excludes_failures_from_means_but_counts_them() -> None:
    """A fragile estimator must not look strong because its failures vanished."""
    summary = aggregate(_raw()).iloc[0]

    assert summary["mise_mean"] == pytest.approx(0.15)
    assert summary["n_replications_attempted"] == 3
    assert summary["n_replications_scored"] == 2


def test_failure_rates_are_reported_separately() -> None:
    rates = failure_rates(_raw()).iloc[0]
    assert rates["attempted"] == 3
    assert rates["fit_failures"] == 1
    assert rates["fit_failure_rate"] == pytest.approx(1 / 3)


def test_completed_cells_supports_resumption(tmp_path) -> None:
    path = tmp_path / "raw.parquet"
    assert completed_cells(path) == set()

    write_raw(_raw().to_dict("records"), path)
    done = completed_cells(path)

    assert ("s", "cox", 0) in done
    assert len(done) == 3


# ---------------------------------------------------------------------------
# Scenario preparation
# ---------------------------------------------------------------------------


def test_tau_excludes_cured_subjects() -> None:
    """A cured subject's recorded time is a sentinel, not a failure time.

    `gen_mixture_cure` writes `max_time * 100` for the cured, which is finite.
    Including it put tau at 1000 for mixture_cure where every other mechanism
    sits between 0.96 and 5.4, so the primary loss was integrated over a
    horizon 385 times too long and that arm was incomparable with the rest.
    """
    from survival_misspec.config import MetricsConfig, ScenarioConfig
    from survival_misspec.experiments import prepare_scenario

    scenario = ScenarioConfig(
        scenario_id="cure",
        dgp="mixture_cure",
        n=500,
        target_censoring=0.5,
        effect_size=0.5,
        params={
            "cure_fraction": 0.3,
            "baseline_hazard": 0.7,
            "betas_survival": [0.5, -0.25],
            "betas_cure": [0.25, 0.1],
            "model_cens": "uniform",
        },
    )
    prepared = prepare_scenario(
        scenario, MetricsConfig(0.8, 51, (0.5,), ("mise",)), calibration_n=4000
    )

    assert prepared.feasible, prepared.reason
    assert prepared.tau < 20.0, (
        f"tau={prepared.tau} looks like the cured sentinel (max_time * 100), "
        "not a failure-time quantile"
    )
