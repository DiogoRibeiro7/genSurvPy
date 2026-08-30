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
    adequacy_region_from_pairs,
    aggregate,
    compact_raw,
    completed_cells,
    failure_rates,
    headline_metric_gap,
    mcse,
    paired_differences,
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
                "root_mean_integrated_squared_error": 0.20,
                "dgp": "cphm",
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 1,
                "fitted": True,
                "scored": True,
                "mise": 0.20,
                "root_mean_integrated_squared_error": 0.30,
                "dgp": "cphm",
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 2,
                "fitted": False,
                "scored": False,
                "mise": np.nan,
                "root_mean_integrated_squared_error": np.nan,
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


def test_aggregate_reports_vector_parameter_recovery() -> None:
    raw = pd.DataFrame(
        [
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 0,
                "fitted": True,
                "scored": True,
                "beta_abs_bias_mean": 0.2,
                "beta_bias_mean": 0.1,
                "beta_rmse": 0.25,
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 1,
                "fitted": True,
                "scored": True,
                "beta_abs_bias_mean": 0.4,
                "beta_bias_mean": 0.3,
                "beta_rmse": 0.45,
            },
        ]
    )

    summary = aggregate(raw).iloc[0]

    assert summary["beta_abs_bias_mean_mean"] == pytest.approx(0.3)
    assert summary["beta_bias_mean_mean"] == pytest.approx(0.2)
    assert summary["beta_rmse_mean"] == pytest.approx(0.35)


def test_paired_differences_preserve_the_matched_design() -> None:
    raw = pd.DataFrame(
        [
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 0,
                "fitted": True,
                "scored": True,
                "mise": 0.10,
            },
            {
                "scenario_id": "s",
                "estimator_id": "rsf",
                "replication_id": 0,
                "fitted": True,
                "scored": True,
                "mise": 0.13,
            },
            {
                "scenario_id": "s",
                "estimator_id": "cox",
                "replication_id": 1,
                "fitted": True,
                "scored": True,
                "mise": 0.20,
            },
            {
                "scenario_id": "s",
                "estimator_id": "rsf",
                "replication_id": 1,
                "fitted": True,
                "scored": True,
                "mise": 0.19,
            },
        ]
    )

    paired = paired_differences(raw, "cox")
    rsf = paired[paired["estimator_id"] == "rsf"].iloc[0]

    assert rsf["mise_difference_mean"] == pytest.approx(0.01)
    assert rsf["mise_difference_mcse"] == pytest.approx(mcse([0.03, -0.01]))

    adequacy = adequacy_region_from_pairs(paired, epsilon=0.02)
    assert bool(adequacy[adequacy["estimator_id"] == "rsf"]["within_epsilon"].iloc[0])


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


def test_completed_cells_requires_the_matching_lock_for_production_resume(
    tmp_path,
) -> None:
    path = tmp_path / "raw.parquet"
    rows = _raw().assign(lock_hash="abc").to_dict("records")
    write_raw(rows, path)

    assert len(completed_cells(path, lock_hash="abc")) == 3
    with pytest.raises(ValueError, match="expected xyz"):
        completed_cells(path, lock_hash="xyz")


def test_completed_cells_refuses_unlocked_rows_for_production_resume(tmp_path) -> None:
    path = tmp_path / "raw.parquet"
    write_raw(_raw().to_dict("records"), path)

    with pytest.raises(ValueError, match="without lock_hash"):
        completed_cells(path, lock_hash="abc")


def test_write_raw_appends_parquet_shards_without_rewriting(tmp_path) -> None:
    path = tmp_path / "raw.parquet"

    write_raw(_raw().to_dict("records"), path)
    write_raw(
        [
            {
                "scenario_id": "s",
                "estimator_id": "rsf",
                "replication_id": 0,
                "fitted": True,
                "scored": True,
                "mise": 0.4,
            }
        ],
        path,
    )

    assert path.is_dir()
    assert len(list(path.glob("*.parquet"))) == 2
    done = completed_cells(path)
    assert ("s", "rsf", 0) in done
    assert len(done) == 4


def test_compact_raw_writes_one_deterministic_file(tmp_path) -> None:
    path = tmp_path / "raw.parquet"
    write_raw(_raw().iloc[[1]].to_dict("records"), path)
    write_raw(_raw().iloc[[0]].to_dict("records"), path)

    compacted = compact_raw(path)
    frame = pd.read_parquet(compacted)

    assert compacted.name == "raw.parquet.compact.parquet"
    assert frame["replication_id"].tolist() == [0, 1]


def test_headline_metric_gap_operationalises_the_primary_claim() -> None:
    summary = pd.DataFrame(
        {
            "c_index_harrell_mean": [0.6, 0.61, 0.7, 0.71],
            "root_mean_integrated_squared_error_mean": [0.05, 0.20, 0.04, 0.30],
            "root_mean_integrated_squared_error_mcse": [0.001] * 4,
        }
    )

    headline = headline_metric_gap(summary, bins=2, quantile=0.9)

    assert list(headline["metric"]) == [
        "c_index_harrell_mean",
        "c_index_harrell_mean",
    ]
    assert headline["loss_quantile"].max() == pytest.approx(0.274)
    assert "loss_quantile_se" in headline.columns
    assert "bootstrap_mc_error" in headline.columns


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
    assert len(prepared.ipcw_time_grid) >= 2


def test_prepared_scenario_rehydrates_from_lock_record() -> None:
    from survival_misspec.config import MetricsConfig, ScenarioConfig
    from survival_misspec.experiments import (
        prepare_scenario,
        prepared_scenario_from_record,
    )

    scenario = ScenarioConfig(
        scenario_id="locked",
        dgp="cphm",
        n=100,
        target_censoring=0.3,
        effect_size=0.5,
        params={"beta": 0.5, "covariate_range": 2.0, "model_cens": "uniform"},
    )
    prepared = prepare_scenario(
        scenario, MetricsConfig(0.8, 21, (0.5,), ("mise",)), calibration_n=1000
    )
    loaded = prepared_scenario_from_record(scenario, prepared.as_record())

    assert loaded.params == prepared.params
    assert loaded.tau == prepared.tau
    assert loaded.time_grid == prepared.time_grid
    assert loaded.ipcw_time_grid == prepared.ipcw_time_grid


def test_ipcw_grid_uses_preparation_support_envelope(monkeypatch) -> None:
    from types import SimpleNamespace

    from survival_misspec import experiments
    from survival_misspec.config import ScenarioConfig

    def fake_draw_replicate(*args, **kwargs):
        return SimpleNamespace(observed_time=np.array([1.5, 5.5]))

    monkeypatch.setattr(experiments, "draw_replicate", fake_draw_replicate)
    scenario = ScenarioConfig(
        scenario_id="ipcw",
        dgp="cphm",
        n=100,
        target_censoring=0.3,
        effect_size=0.5,
        params={},
    )

    grid = experiments.ipcw_support_grid(
        scenario,
        {"cens_par": 1.0},
        (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        support_replications=3,
        support_target=0.95,
    )

    assert grid == (3.0, 4.0)


def test_cells_are_independent_of_the_order_they_are_run_in() -> None:
    """The property that makes parallel execution the same experiment.

    Running cells in a different order must give byte-identical rows. This is
    what licenses `--workers N`: a cell's data depends only on its identifiers,
    never on which worker ran it, how many workers there were, or when it
    finished. Verified against a real 4-worker run as well: seeds, MISE,
    concordance, integrated Brier and calibration error were all bit-identical
    to the sequential run.
    """
    from survival_misspec.config import EstimatorConfig, MetricsConfig, ScenarioConfig
    from survival_misspec.experiments import prepare_scenario, run_cell

    scenario = ScenarioConfig(
        scenario_id="order",
        dgp="cphm",
        n=150,
        target_censoring=0.3,
        effect_size=0.5,
        params={"beta": 0.5, "covariate_range": 2.0, "model_cens": "uniform"},
    )
    prepared = prepare_scenario(
        scenario, MetricsConfig(0.8, 21, (0.5,), ("mise",)), calibration_n=2000
    )
    estimator = EstimatorConfig("cox_ph", "cox_ph")

    forwards = [run_cell(prepared, estimator, r, 7) for r in (0, 1, 2)]
    backwards = [run_cell(prepared, estimator, r, 7) for r in (2, 1, 0)][::-1]

    for first, second in zip(forwards, backwards):
        assert first["train_seed"] == second["train_seed"]
        assert first["eval_seed"] == second["eval_seed"]
        assert first["mise"] == second["mise"]
        assert first["c_index_harrell"] == second["c_index_harrell"]
