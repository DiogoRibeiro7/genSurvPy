"""The sanity condition that gates the whole study.

Section 17 of the protocol: if the pipeline cannot recover a known parameter
under a correctly specified model, nothing it reports about *mis*specification
can be believed, because a bias in the plumbing is indistinguishable from a
bias attributed to the estimator.

So before any production run, a large correctly specified Cox simulation must
give

.. math::

    \\hat\\beta \\rightarrow \\beta.

This is deliberately a strong, slow test. It is the one that says the machinery
works end to end -- generator, truth, adapter, and the correspondence between a
fitted coefficient and a DGP parameter.
"""

from __future__ import annotations

import numpy as np
import pytest
from survival_misspec.estimators import fit_estimator
from survival_misspec.experiments import PARAMETER_CORRESPONDENCE, parameter_recovery
from survival_misspec.simulation import draw_replicate


@pytest.mark.slow
@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_cox_recovers_the_cphm_coefficient(beta: float) -> None:
    """The correctly specified case, at a sample size where bias would show."""
    params = {
        "beta": beta,
        "covariate_range": 2.0,
        "model_cens": "uniform",
        "cens_par": 2.0,
    }

    estimates = []
    for replication in range(12):
        replicate = draw_replicate(
            "cphm", params, 4000, "recovery", replication, 20260828
        )
        fitted = fit_estimator(
            "cox_ph",
            "cox_ph",
            {},
            replicate.covariates,
            replicate.observed_time,
            replicate.event,
        )
        assert fitted.fitted, fitted.error
        estimates.append(fitted.model.coefficients()["X0"])

    estimates = np.asarray(estimates, dtype=float)
    bias = estimates.mean() - beta
    standard_error = estimates.std(ddof=1) / np.sqrt(estimates.size)

    assert abs(bias) < 4 * standard_error + 0.02, (
        f"Cox did not recover beta={beta}: mean estimate {estimates.mean():.4f}, "
        f"bias {bias:+.4f}, MCSE {standard_error:.4f}. The pipeline cannot be "
        f"trusted to measure misspecification until this passes."
    )


@pytest.mark.slow
def test_cox_recovers_the_piecewise_exponential_coefficients() -> None:
    """Proportional hazards with a baseline no parametric model here can match.

    Cox is correctly specified for the *coefficients* even though the baseline
    is a step function, which is precisely why it belongs in the correspondence
    table.
    """
    betas = [0.6, -0.3]
    params = {
        "betas": betas,
        "breakpoints": [0.5, 1.5],
        "hazard_rates": [0.4, 0.9, 1.6],
        "model_cens": "uniform",
        "cens_par": 3.0,
    }

    estimates = []
    for replication in range(10):
        replicate = draw_replicate(
            "piecewise_exponential", params, 4000, "recovery_pw", replication, 20260828
        )
        fitted = fit_estimator(
            "cox_ph",
            "cox_ph",
            {},
            replicate.covariates,
            replicate.observed_time,
            replicate.event,
        )
        assert fitted.fitted, fitted.error
        coefficients = fitted.model.coefficients()
        estimates.append([coefficients["X0"], coefficients["X1"]])

    matrix = np.asarray(estimates, dtype=float)
    for index, true_value in enumerate(betas):
        column = matrix[:, index]
        bias = column.mean() - true_value
        standard_error = column.std(ddof=1) / np.sqrt(column.size)
        assert abs(bias) < 4 * standard_error + 0.03, (
            f"beta[{index}]: mean {column.mean():.4f} against true {true_value}, "
            f"bias {bias:+.4f}, MCSE {standard_error:.4f}"
        )


def test_parameter_recovery_is_declined_where_the_estimand_differs() -> None:
    """A Weibull AFT coefficient is not an estimate of a log-normal beta.

    Reporting a "bias" there would subtract an acceleration factor from a log
    hazard ratio and call the difference an error. The correspondence table is
    deliberately sparse, and this pins that it stays so.
    """
    outcome = parameter_recovery(
        "aft_ln", "weibull_aft", "weibull_aft", {"X0": 0.4}, {"beta": [0.5]}
    )
    assert outcome == {"parameter_recovery_applicable": False}


def test_parameter_recovery_reports_bias_where_the_estimand_matches() -> None:
    outcome = parameter_recovery(
        "cphm", "cox_ph", "cox_ph", {"X0": 0.62}, {"beta": 0.5}
    )

    assert outcome["parameter_recovery_applicable"]
    assert outcome["beta_bias_scalar"] == pytest.approx(0.12)
    assert outcome["beta_bias_mean"] == pytest.approx(0.12)
    assert outcome["beta_abs_bias_mean"] == pytest.approx(0.12)
    assert outcome["beta_rmse"] == pytest.approx(0.12)
    assert outcome["beta_true_scalar"] == pytest.approx(0.5)


def test_parameter_recovery_reports_vector_bias_summaries() -> None:
    outcome = parameter_recovery(
        "piecewise_exponential",
        "cox_ph",
        "cox_ph",
        {"X0": 0.7, "X1": -0.1},
        {"betas": [0.5, -0.3]},
    )

    assert outcome["parameter_recovery_applicable"]
    assert outcome["beta_bias"] == pytest.approx([0.2, 0.2])
    assert outcome["beta_bias_mean"] == pytest.approx(0.2)
    assert outcome["beta_abs_bias_mean"] == pytest.approx(0.2)
    assert outcome["beta_rmse"] == pytest.approx(0.2)


def test_the_correspondence_table_only_lists_proportional_hazards_cases() -> None:
    """Every entry must be a mechanism where a Cox coefficient estimates beta."""
    proportional = {"cphm", "aft_weibull", "piecewise_exponential"}
    for dgp, adapter in PARAMETER_CORRESPONDENCE:
        assert dgp in proportional, (
            f"{dgp} is not a proportional-hazards mechanism, so a Cox "
            f"coefficient does not estimate its beta"
        )
        assert adapter == "cox_ph"
