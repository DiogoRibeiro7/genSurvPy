"""Every generator must reject NaN and infinity in every numeric argument.

Comparisons with NaN are all false, so a check written as ``value <= 0``
silently admits it, and ``inf > 0`` is true. Both then reach NumPy. What came
back was one of three things, none of them an error the caller could act on:

- a frame of the right shape, quietly full of NaN;
- ``OverflowError: high - low range exceeds valid bounds`` from a uniform draw,
  naming nothing the caller passed;
- for ``gen_recurrent_events(followup_time=nan)``, a call that never returned,
  because the sampling loop compares each candidate event time against a bound
  that no value can exceed.

Thirty-nine arguments across the twelve generators were affected. These tests
walk every numeric argument of every model and require a ``ValidationError``.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from gen_surv import generate
from gen_surv.validation import (
    ParameterError,
    PositiveValueError,
    ValidationError,
    ensure_finite,
    ensure_positive,
)

# A valid call for every registered model, kept small: these tests are about
# rejection, so nothing needs to be large.
VALID_CALLS: dict[str, dict[str, Any]] = {
    "cphm": dict(
        n=20, beta=0.5, covariate_range=2.0, model_cens="uniform", cens_par=1.0
    ),
    "aft_ln": dict(n=20, beta=[0.5], sigma=1.0, model_cens="uniform", cens_par=1.0),
    "aft_weibull": dict(
        n=20, beta=[0.5], shape=1.5, scale=2.0, model_cens="uniform", cens_par=1.0
    ),
    "aft_log_logistic": dict(
        n=20, beta=[0.5], shape=1.5, scale=2.0, model_cens="uniform", cens_par=1.0
    ),
    "piecewise_exponential": dict(
        n=20, breakpoints=[1.0], hazard_rates=[0.5, 1.0], betas=[0.1, 0.1]
    ),
    "competing_risks": dict(
        n=20, n_risks=2, baseline_hazards=[0.4, 0.2], betas=[[0.1, 0.0], [0.0, 0.1]]
    ),
    "competing_risks_weibull": dict(
        n=20, n_risks=2, shape_params=[1.2, 0.8], scale_params=[2.0, 1.5]
    ),
    "mixture_cure": dict(
        n=20,
        cure_fraction=0.3,
        baseline_hazard=0.5,
        betas_survival=[0.1, 0.1],
        betas_cure=[0.1, 0.1],
    ),
    "cmm": dict(
        n=20,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
    ),
    "thmm": dict(
        n=20,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.1, 0.2, 0.3],
        covariate_range=1.0,
        rate=[0.2, 0.3, 0.4],
    ),
    "tdcm": dict(
        n=20,
        dist="weibull",
        corr=0.5,
        dist_par=[1.0, 2.0, 1.0, 2.0],
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.5, 0.3],
        lam=1.0,
    ),
    "recurrent_events": dict(
        n=20,
        baseline_params={"rate": 0.5},
        betas=[0.1, 0.1],
        followup_time=5.0,
        cens_par=8.0,
    ),
}

HOSTILE = [float("nan"), float("inf"), float("-inf")]


def _scalar_arguments(call: dict[str, Any]) -> list[str]:
    return [
        key
        for key, value in call.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and key != "n"
    ]


def _sequence_arguments(call: dict[str, Any]) -> list[str]:
    return [key for key, value in call.items() if isinstance(value, list) and value]


def test_every_model_has_a_valid_call_here() -> None:
    from gen_surv.interface import _model_map

    assert set(VALID_CALLS) == set(_model_map), "a model was added without a case"


@pytest.mark.parametrize("model", sorted(VALID_CALLS))
def test_valid_call_still_works(model: str) -> None:
    """The baseline each hostile case is derived from must itself be valid."""
    assert len(generate(model=model, **VALID_CALLS[model], seed=1)) > 0


@pytest.mark.parametrize("model", sorted(VALID_CALLS))
def test_non_finite_scalars_are_rejected(model: str) -> None:
    call = VALID_CALLS[model]

    for argument in _scalar_arguments(call):
        for hostile in HOSTILE:
            kwargs = {**call, argument: hostile, "seed": 1}
            with pytest.raises(ValidationError):
                generate(model=model, **kwargs)


@pytest.mark.parametrize("model", sorted(VALID_CALLS))
def test_non_finite_sequence_entries_are_rejected(model: str) -> None:
    call = VALID_CALLS[model]

    for argument in _sequence_arguments(call):
        for hostile in (float("nan"), float("inf")):
            poisoned = list(call[argument])
            if isinstance(poisoned[0], list):
                poisoned[0] = [hostile, *poisoned[0][1:]]
            else:
                poisoned[0] = hostile

            kwargs = {**call, argument: poisoned, "seed": 1}
            with pytest.raises(ValidationError):
                generate(model=model, **kwargs)


@pytest.mark.parametrize("model", ["cmm", "thmm"])
def test_negative_transition_rates_are_rejected(model: str) -> None:
    """Only the length of ``rate`` was checked.

    A negative entry reached NumPy and surfaced as ``ValueError: scale < 0``
    from inside the random generator, naming nothing the caller had passed.
    """
    call = dict(VALID_CALLS[model])
    call["rate"] = [
        -abs(value) if index == 0 else value for index, value in enumerate(call["rate"])
    ]

    with pytest.raises(ValidationError, match="rate"):
        generate(model=model, **call, seed=1)


def test_recurrent_events_with_a_nan_horizon_returns(monkeypatch) -> None:
    """This call used to hang rather than raise.

    ``followup_time=nan`` made every ``candidate >= end`` comparison false, so
    the sampling loop had no exit and ran until the process was killed.
    """
    with pytest.raises(ValidationError):
        generate(
            model="recurrent_events",
            n=5,
            followup_time=float("nan"),
            betas=[0.1, 0.1],
            seed=1,
        )


# --------------------------------------------------------------------------
# The helpers themselves
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ensure_positive_rejects_non_finite(value: float) -> None:
    with pytest.raises(ParameterError, match="finite"):
        ensure_positive(value, "value")


def test_ensure_positive_keeps_its_existing_contract() -> None:
    """A bool or a non-number is still a PositiveValueError, as before."""
    ensure_positive(0.1, "value")

    with pytest.raises(PositiveValueError):
        ensure_positive(True, "value")
    with pytest.raises(PositiveValueError):
        ensure_positive(-1.0, "value")
    with pytest.raises(PositiveValueError):
        ensure_positive(0.0, "value")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_ensure_finite_rejects_non_finite(value: float) -> None:
    with pytest.raises(ParameterError, match="finite"):
        ensure_finite(value, "value")


def test_ensure_finite_accepts_any_finite_sign() -> None:
    for value in (-3.5, 0.0, 2.75):
        ensure_finite(value, "value")
        assert math.isfinite(value)


def test_ensure_finite_rejects_a_bool_as_a_number() -> None:
    with pytest.raises(ParameterError, match="number"):
        ensure_finite(True, "value")
