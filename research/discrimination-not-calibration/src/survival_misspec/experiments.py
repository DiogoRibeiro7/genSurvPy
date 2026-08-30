"""Running one cell of the design, and preparing the scenarios that define them.

A *cell* is one ``(scenario, estimator, replication)`` triple. Running it means:
draw a training sample and an independent evaluation sample, fit, predict,
score against the known truth, and return one flat row. Everything the row
needs to be traced back to its origin travels with it.

Scenario preparation is separate and happens once. Two quantities have to be
resolved before any replication runs and then held fixed:

``cens_par``
    Calibrated to the target censoring rate. Recalibrating per replicate would
    make the censoring mechanism itself random, so replicates would no longer
    be draws from one scenario.

``tau``
    The integration horizon, set as a quantile of the *true* marginal
    event-time distribution. Time scales differ by orders of magnitude between
    these generators -- a horizon of 1.0 is most of the support for one and a
    rounding error for another -- so a fixed number would not mean the same
    thing twice. Deriving it from the truth rather than from the observed data
    also keeps it independent of the censoring level, which is a design factor
    in its own right and must not move the target.

Both are computed on a dedicated calibration seed, unrelated to any
replication, and recorded in the experiment lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from gen_surv import simulate as gen_surv_simulate

from .config import EstimatorConfig, MetricsConfig, ScenarioConfig
from .estimators import fit_estimator
from .metrics import evaluate_all
from .simulation import calibrate_censoring, draw_replicate
from .truth import true_survival

__all__ = [
    "PreparedScenario",
    "prepare_scenario",
    "prepared_scenario_from_record",
    "ipcw_support_grid",
    "PARAMETER_CORRESPONDENCE",
    "parameter_recovery",
    "run_cell",
]

#: Size of the independent evaluation sample. Fixed across scenarios so that
#: the Monte Carlo error of the truth-based losses does not vary with the
#: training size, which is a design factor. Large enough that the evaluation
#: contributes little to the total variance of MISE across replications.
EVALUATION_N = 4000

#: Seed for scenario preparation only. Deliberately unrelated to any
#: replication seed, so preparation cannot correlate with the data it defines.
PREPARATION_SEED = 314159265

#: Number of preparation-only matched train/evaluation draws used to place the
#: scenario-level IPCW grid inside ordinary observed follow-up support. These
#: draws are not production replications and use a separate seed stream.
IPCW_SUPPORT_REPLICATIONS = 100

#: Preparation support envelope used when fixing the IPCW grid before
#: production. The acceptance gate remains 0.95; using the strict envelope from
#: preparation draws gives the audit room for finite-sample tail noise.
IPCW_SUPPORT_TARGET = 1.0

#: Additional guard against placing a fixed IPCW point exactly next to the
#: prospective support boundary. Only applied when enough grid points remain.
IPCW_SUPPORT_TRIM_POINTS = 1


@dataclass(frozen=True)
class PreparedScenario:
    """A scenario with its calibrated censoring and fixed evaluation horizon."""

    config: ScenarioConfig
    params: Mapping[str, Any]
    tau: float
    time_grid: tuple[float, ...]
    ipcw_time_grid: tuple[float, ...]
    censoring_achieved: float
    feasible: bool
    reason: str = ""

    @property
    def scenario_id(self) -> str:
        return self.config.scenario_id

    def as_record(self) -> dict[str, Any]:
        """Flat form for the experiment lock and for provenance columns."""
        return {
            "scenario_id": self.scenario_id,
            "dgp": self.config.dgp,
            "n": self.config.n,
            "target_censoring": self.config.target_censoring,
            "achieved_censoring": self.censoring_achieved,
            "effect_size": self.config.effect_size,
            "misspecification": self.config.misspecification,
            "tau": self.tau,
            "n_time_points": len(self.time_grid),
            "time_grid": list(self.time_grid),
            "ipcw_time_grid": list(self.ipcw_time_grid),
            "params": dict(self.params),
            "feasible": self.feasible,
            "reason": self.reason,
            "scenario_hash": self.config.hash,
        }


def ipcw_support_grid(
    scenario: ScenarioConfig,
    params: Mapping[str, Any],
    grid: tuple[float, ...],
    *,
    support_replications: int = IPCW_SUPPORT_REPLICATIONS,
    master_seed: int = PREPARATION_SEED,
    evaluation_n: int = EVALUATION_N,
    support_target: float = IPCW_SUPPORT_TARGET,
    trim_points: int = IPCW_SUPPORT_TRIM_POINTS,
) -> tuple[float, ...]:
    """A fixed IPCW grid with prospective train/evaluation support.

    Brier and time-dependent AUC require their time grid to lie inside the
    observed support of both the training and evaluation samples. Instead of
    taking the maximum observed time from a single large preparation draw, use
    matched preparation-only draws to choose a conservative common-support
    interval, then keep the ordinary tau-based grid points that fall inside it.
    """
    if support_replications <= 0:
        return tuple(value for value in grid if value > 0.0)
    if not 0.0 < support_target <= 1.0:
        raise ValueError("support_target must be in (0, 1]")

    lower_bounds = []
    upper_bounds = []
    for replication_id in range(support_replications):
        train = draw_replicate(
            scenario.dgp,
            params,
            scenario.n,
            scenario.scenario_id,
            replication_id,
            master_seed,
            stream="ipcw_train",
        )
        evaluation = draw_replicate(
            scenario.dgp,
            params,
            evaluation_n,
            scenario.scenario_id,
            replication_id,
            master_seed,
            stream="ipcw_eval",
        )
        lower_bounds.append(
            max(
                float(np.min(train.observed_time)),
                float(np.min(evaluation.observed_time)),
            )
        )
        upper_bounds.append(
            min(
                float(np.max(train.observed_time)),
                float(np.max(evaluation.observed_time)),
            )
        )

    if support_target == 1.0:
        lower = max(lower_bounds)
        upper = min(upper_bounds)
    else:
        lower = float(np.quantile(lower_bounds, support_target))
        upper = float(np.quantile(upper_bounds, 1.0 - support_target))
    supported = tuple(value for value in grid if value > lower and value < upper)
    if trim_points > 0 and len(supported) > 2 * trim_points:
        return supported[trim_points:-trim_points]
    return supported


def prepare_scenario(
    scenario: ScenarioConfig,
    metrics: MetricsConfig,
    *,
    calibration_n: int = 20000,
    ipcw_support_replications: int = IPCW_SUPPORT_REPLICATIONS,
) -> PreparedScenario:
    """Resolve censoring and the evaluation horizon, once, before anything runs."""
    calibration = calibrate_censoring(
        scenario.dgp,
        scenario.params,
        scenario.target_censoring,
        n=calibration_n,
        seed=PREPARATION_SEED,
    )
    params = {**scenario.params, "cens_par": calibration.cens_par}

    # tau from the *latent* event times, so it does not move with censoring.
    reference = gen_surv_simulate(
        scenario.dgp, n=calibration_n, **params, seed=PREPARATION_SEED + 1
    )
    latent = np.asarray(reference.truth["event_time"], dtype=float)
    usable = np.isfinite(latent)

    # A cured subject never fails, and `gen_mixture_cure` encodes that as
    # `max_time * 100` -- a finite sentinel, not an event time. Taking a
    # quantile over it put tau at 1000 for mixture_cure where every other
    # mechanism sits between 0.96 and 5.4, so MISE was integrated over a
    # horizon 385 times too long and that arm of the study was not comparable
    # with any other. The truth records cure status, so exclude it explicitly
    # rather than trying to recognise the sentinel by its value.
    cured = reference.truth.get("cured")
    if cured is not None:
        usable &= np.asarray(cured) == 0

    finite = latent[usable]

    if finite.size == 0:
        return PreparedScenario(
            config=scenario,
            params=params,
            tau=float("nan"),
            time_grid=(),
            ipcw_time_grid=(),
            censoring_achieved=calibration.achieved,
            feasible=False,
            reason="no finite latent event times; tau is undefined",
        )

    tau = float(np.quantile(finite, metrics.tau_quantile))

    # tau should sit inside the bulk of the failure times. If it is orders of
    # magnitude above the median, something non-failure has been counted as an
    # event time -- which is how the sentinel above was found.
    median = float(np.median(finite))
    if median > 0 and tau > 50 * median:
        return PreparedScenario(
            config=scenario,
            params=params,
            tau=tau,
            time_grid=(),
            ipcw_time_grid=(),
            censoring_achieved=calibration.achieved,
            feasible=False,
            reason=(
                f"tau={tau:.4g} is more than 50x the median failure time "
                f"({median:.4g}); a non-failure value is probably being counted "
                f"as an event time"
            ),
        )

    grid = tuple(np.linspace(0.0, tau, metrics.n_time_points).tolist())
    ipcw_grid = ipcw_support_grid(
        scenario,
        params,
        grid,
        support_replications=ipcw_support_replications,
    )

    return PreparedScenario(
        config=scenario,
        params=params,
        tau=tau,
        time_grid=grid,
        ipcw_time_grid=ipcw_grid,
        censoring_achieved=calibration.achieved,
        feasible=calibration.feasible,
        reason=calibration.reason,
    )


def prepared_scenario_from_record(
    scenario: ScenarioConfig, record: Mapping[str, Any]
) -> PreparedScenario:
    """Rehydrate a prepared scenario from the experiment lock.

    Production runs must not recalibrate censoring or recompute tau when they
    resume. The lock contains the prepared values; this function turns them back
    into the object `run_cell` expects and refuses stale or incomplete locks.
    """
    if record.get("scenario_id") != scenario.scenario_id:
        raise ValueError(
            f"lock record for {record.get('scenario_id')!r} cannot prepare "
            f"scenario {scenario.scenario_id!r}"
        )
    if record.get("scenario_hash") != scenario.hash:
        raise ValueError(
            f"scenario {scenario.scenario_id!r} has changed since the lock was written"
        )

    required = ("params", "tau", "time_grid", "ipcw_time_grid", "feasible")
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(
            f"prepared scenario {scenario.scenario_id!r} is missing {missing}"
        )

    return PreparedScenario(
        config=scenario,
        params=dict(record["params"]),
        tau=float(record["tau"]),
        time_grid=tuple(float(value) for value in record["time_grid"]),
        ipcw_time_grid=tuple(float(value) for value in record["ipcw_time_grid"]),
        censoring_achieved=float(
            record.get("achieved_censoring", record.get("censoring_achieved", np.nan))
        ),
        feasible=bool(record["feasible"]),
        reason=str(record.get("reason", "")),
    )


#: Where a fitted coefficient corresponds *exactly* to a DGP parameter, so
#: bias and coverage are meaningful. Deliberately sparse: a Cox coefficient
#: estimates the log hazard ratio, which equals the generator's ``beta`` only
#: when the mechanism is proportional hazards. Everywhere else the estimand
#: differs and reporting "bias" would compare two different quantities.
PARAMETER_CORRESPONDENCE: dict[tuple[str, str], str] = {
    ("cphm", "cox_ph"): "beta",
    ("aft_weibull", "cox_ph"): "betas",
    ("piecewise_exponential", "cox_ph"): "betas",
}


def parameter_recovery(
    dgp: str,
    estimator_id: str,
    adapter: str,
    coefficients: Mapping[str, float] | None,
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Bias of the fitted coefficients, where a true counterpart exists.

    Returns ``applicable=False`` rather than a number when the estimator has no
    parameter corresponding to the DGP's. Reporting a "bias" for a Weibull AFT
    coefficient against a log-normal ``beta`` would compare an acceleration
    factor with a log hazard ratio -- a different estimand, not a worse
    estimate of the same one.
    """
    key = PARAMETER_CORRESPONDENCE.get((dgp, adapter))
    if key is None or coefficients is None:
        return {"parameter_recovery_applicable": False}

    truth_value = params.get(key)
    if truth_value is None:
        return {"parameter_recovery_applicable": False}

    true_vector = np.atleast_1d(np.asarray(truth_value, dtype=float))
    fitted = np.array(
        [coefficients.get(f"X{i}", np.nan) for i in range(true_vector.size)],
        dtype=float,
    )

    bias = fitted - true_vector
    out: dict[str, Any] = {
        "parameter_recovery_applicable": True,
        "beta_true": true_vector.tolist(),
        "beta_hat": fitted.tolist(),
        "beta_bias": bias.tolist(),
        "beta_bias_mean": float(np.mean(bias)),
        "beta_abs_bias_mean": float(np.mean(np.abs(bias))),
        "beta_rmse": float(np.sqrt(np.mean(bias**2))),
    }
    # A single-covariate scenario is the common case and is far easier to
    # aggregate if the scalar is also available directly.
    if true_vector.size == 1:
        out.update(
            {
                "beta_true_scalar": float(true_vector[0]),
                "beta_hat_scalar": float(fitted[0]),
                "beta_bias_scalar": float(bias[0]),
            }
        )
    return out


def run_cell(
    prepared: PreparedScenario,
    estimator: EstimatorConfig,
    replication_id: int,
    master_seed: int,
    *,
    evaluation_n: int = EVALUATION_N,
) -> dict[str, Any]:
    """Run one ``(scenario, estimator, replication)`` triple to a single row.

    Never raises for a modelling failure. A row is always returned, with
    ``fitted=False`` and the error recorded, because the rate at which an
    estimator fails in a regime is part of its performance in that regime and
    dropping those rows would flatter it.
    """
    scenario = prepared.config
    row: dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "replication_id": replication_id,
        "estimator_id": estimator.estimator_id,
        "adapter": estimator.adapter,
        "dgp": scenario.dgp,
        "n": scenario.n,
        "target_censoring": scenario.target_censoring,
        "effect_size": scenario.effect_size,
        "misspecification": scenario.misspecification,
        "tau": prepared.tau,
        "scenario_hash": scenario.hash,
        "estimator_hash": estimator.hash,
    }

    train = draw_replicate(
        scenario.dgp,
        prepared.params,
        scenario.n,
        scenario.scenario_id,
        replication_id,
        master_seed,
        stream="train",
    )
    evaluation = draw_replicate(
        scenario.dgp,
        prepared.params,
        evaluation_n,
        scenario.scenario_id,
        replication_id,
        master_seed,
        stream="eval",
    )

    row.update(
        {
            "train_seed": train.seed,
            "eval_seed": evaluation.seed,
            "realised_censoring": train.censoring_rate,
            "n_events_train": int(train.event.sum()),
        }
    )

    fitted = fit_estimator(
        estimator.estimator_id,
        estimator.adapter,
        estimator.params,
        train.covariates,
        train.observed_time,
        train.event,
    )
    row.update(
        {
            "fitted": fitted.fitted,
            "fit_error_type": fitted.error_type,
            "fit_error": fitted.error[:500],
            "fit_warnings": "; ".join(fitted.warnings)[:500],
            "fit_runtime_seconds": fitted.runtime_seconds,
        }
    )

    if not fitted.fitted:
        return row

    grid = np.asarray(prepared.time_grid, dtype=float)
    try:
        predicted = fitted.model.predict_survival(evaluation.covariates, grid)
        risk = fitted.model.predict_risk(evaluation.covariates)
        truth_surface = true_survival(
            scenario.dgp, grid, evaluation.truth, prepared.params
        )
        row.update(
            evaluate_all(
                predicted=predicted,
                truth=truth_surface,
                risk=risk,
                times=grid,
                tau=prepared.tau,
                train_time=train.observed_time,
                train_event=train.event,
                eval_time=evaluation.observed_time,
                eval_event=evaluation.event,
                prediction_error_times=np.asarray(prepared.ipcw_time_grid, dtype=float),
                survival_at_times=lambda requested_times: (
                    fitted.model.predict_survival_at_times(
                        evaluation.covariates, requested_times
                    )
                ),
            )
        )
        row["scored"] = True
    except Exception as exception:  # noqa: BLE001 - scoring failure is a result
        row["scored"] = False
        row["score_error_type"] = type(exception).__name__
        row["score_error"] = f"{exception}"[:500]
        return row

    row.update(
        parameter_recovery(
            scenario.dgp,
            estimator.estimator_id,
            estimator.adapter,
            fitted.model.coefficients(),
            prepared.params,
        )
    )
    return row
