"""Invariants that must hold for every generator, over parameters Hypothesis picks.

``test_distributions.py`` asks whether one model's sampled times follow the
distribution it declares. These tests ask a weaker question of all twelve at
once: whatever the parameters, is the frame that comes back well formed, and
does the same seed still give the same data?

A weaker question asked over a wider surface finds a different class of defect.
The fixed-parameter suites exercise one point per model. These walk a range, so
a bug that only shows at a small ``n``, a large hazard, a coefficient near zero
or a censoring distribution that swallows every event has somewhere to appear.

Three families here:

- **Output invariants.** The column contract, no NaN or infinity, ``status`` in
  ``{0, 1}``, times non-negative, intervals never inverted, states within the
  declared set.
- **The seed contract.** Documented on the reproducibility page: the same seed
  gives the same frame, and an ``int`` agrees with the generator it seeds. Both
  are promises to users, and neither was tested across all twelve models.
- **Rejection.** A value outside a parameter's domain must raise
  ``ValidationError``, naming the argument -- never return a frame, and never
  surface as a NumPy error about something the caller did not pass.

``test_input_hardening.py`` covers NaN and infinity exhaustively at fixed
points; this covers the ordinary out-of-domain values over a range.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gen_surv import generate
from gen_surv.interface import _model_map
from gen_surv.validation import ValidationError
from tests.test_generate_regression import EXPECTED_COLUMNS

# Keep every example small and quick: these tests are about the shape of the
# contract, not statistical power, and Hypothesis runs each one many times.
SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)

N = st.integers(min_value=2, max_value=25)
SEED = st.integers(min_value=0, max_value=10_000)
CENS = st.sampled_from(["uniform", "exponential"])
POSITIVE = st.floats(
    min_value=0.05, max_value=4.0, allow_nan=False, allow_infinity=False
)
COEF = st.floats(min_value=-1.5, max_value=1.5, allow_nan=False, allow_infinity=False)
FRACTION = st.floats(
    min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False
)


def _coefs(size: int) -> st.SearchStrategy[list]:
    return st.lists(COEF, min_size=size, max_size=size)


def _positives(size: int) -> st.SearchStrategy[list]:
    return st.lists(POSITIVE, min_size=size, max_size=size)


#: `dist_par` for `tdcm` is drawn from a narrower band than POSITIVE. The
#: bivariate sampler builds each margin as `(-log(1 - u) / a) ** (1 / b)`, so a
#: `b` well below 1 is an exponent well above 1 and the covariate reaches the
#: tens of thousands. `exp(beta[0] * z)` then leaves the range of a float,
#: which `gen_tdcm` now refuses -- see
#: `test_tdcm_raises_rather_than_emitting_zero_time_events` for what it used to
#: return instead. Excluded here so the invariant tests stay on parameters that
#: are meant to produce a frame; the excluded region has its own test.
MODERATE = st.floats(
    min_value=0.5, max_value=4.0, allow_nan=False, allow_infinity=False
)


@st.composite
def _tdcm_kwargs(draw: st.DrawFn) -> Dict[str, Any]:
    """``dist`` constrains both ``corr`` and the length of ``dist_par``."""
    dist = draw(st.sampled_from(["weibull", "exponential"]))
    if dist == "weibull":
        corr = draw(
            st.floats(
                min_value=0.01,
                max_value=1.0,
                exclude_max=True,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        dist_par = draw(st.lists(MODERATE, min_size=4, max_size=4))
    else:
        corr = draw(
            st.floats(
                min_value=-1.0,
                max_value=1.0,
                exclude_min=True,
                exclude_max=True,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        dist_par = draw(st.lists(MODERATE, min_size=2, max_size=2))
    return {
        "n": draw(N),
        "dist": dist,
        "corr": corr,
        "dist_par": dist_par,
        "model_cens": draw(CENS),
        "cens_par": draw(POSITIVE),
        "beta": draw(_coefs(2)),
        "lam": draw(POSITIVE),
    }


MODEL_STRATEGIES: Dict[str, st.SearchStrategy[Dict[str, Any]]] = {
    "cphm": st.fixed_dictionaries(
        {
            "n": N,
            "beta": COEF,
            "covariate_range": POSITIVE,
            "model_cens": CENS,
            "cens_par": POSITIVE,
        }
    ),
    "aft_ln": st.fixed_dictionaries(
        {
            "n": N,
            "beta": _coefs(1),
            "sigma": POSITIVE,
            "model_cens": CENS,
            "cens_par": POSITIVE,
        }
    ),
    "aft_weibull": st.fixed_dictionaries(
        {
            "n": N,
            "beta": _coefs(1),
            "shape": POSITIVE,
            "scale": POSITIVE,
            "model_cens": CENS,
            "cens_par": POSITIVE,
        }
    ),
    "aft_log_logistic": st.fixed_dictionaries(
        {
            "n": N,
            "beta": _coefs(1),
            "shape": POSITIVE,
            "scale": POSITIVE,
            "model_cens": CENS,
            "cens_par": POSITIVE,
        }
    ),
    "piecewise_exponential": st.fixed_dictionaries(
        {
            "n": N,
            "breakpoints": _positives(1),
            "hazard_rates": _positives(2),
            "betas": _coefs(2),
        }
    ),
    "competing_risks": st.fixed_dictionaries(
        {
            "n": N,
            "n_risks": st.just(2),
            "baseline_hazards": _positives(2),
            "betas": st.lists(_coefs(2), min_size=2, max_size=2),
        }
    ),
    "competing_risks_weibull": st.fixed_dictionaries(
        {
            "n": N,
            "n_risks": st.just(2),
            "shape_params": _positives(2),
            "scale_params": _positives(2),
        }
    ),
    "mixture_cure": st.fixed_dictionaries(
        {
            "n": N,
            "cure_fraction": FRACTION,
            "baseline_hazard": POSITIVE,
            "betas_survival": _coefs(2),
            "betas_cure": _coefs(2),
        }
    ),
    "cmm": st.fixed_dictionaries(
        {
            "n": N,
            "model_cens": CENS,
            "cens_par": POSITIVE,
            "beta": _coefs(3),
            "covariate_range": POSITIVE,
            "rate": _positives(6),
        }
    ),
    "thmm": st.fixed_dictionaries(
        {
            "n": N,
            "model_cens": CENS,
            "cens_par": POSITIVE,
            "beta": _coefs(3),
            "covariate_range": POSITIVE,
            "rate": _positives(3),
        }
    ),
    "tdcm": _tdcm_kwargs(),
    "recurrent_events": st.fixed_dictionaries(
        {
            "n": N,
            "baseline_params": st.fixed_dictionaries({"rate": POSITIVE}),
            "betas": _coefs(2),
            "followup_time": POSITIVE,
            "cens_par": POSITIVE,
        }
    ),
}

#: Models returning exactly one row per subject. The rest return a variable
#: number, so only the subject count can be checked.
ONE_ROW_PER_SUBJECT = {
    "cphm",
    "aft_ln",
    "aft_weibull",
    "aft_log_logistic",
    "piecewise_exponential",
    "competing_risks",
    "competing_risks_weibull",
    "mixture_cure",
    "tdcm",
}

#: Frames laid out as counting-process risk intervals.
INTERVAL_LAYOUT = {"cmm", "tdcm", "recurrent_events"}


def test_every_registered_model_has_a_strategy() -> None:
    """A model added without a strategy would silently escape all of this."""
    assert set(MODEL_STRATEGIES) == set(_model_map)


def _assert_invariants(model: str, df: pd.DataFrame, kwargs: Dict[str, Any]) -> None:
    n = kwargs["n"]

    assert list(df.columns) == EXPECTED_COLUMNS[model], f"{model} broke its columns"
    assert len(df) > 0, f"{model} returned an empty frame"

    numeric = df.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all(), f"{model} produced NaN or infinity"

    if "status" in df.columns:
        # Competing risks put the winning cause in `status`, not an indicator:
        # 0 is censored and 1..n_risks name the cause. Every other model is
        # binary.
        allowed = set(range(kwargs.get("n_risks", 1) + 1))
        assert (
            set(df["status"].unique()) <= allowed
        ), f"{model} has a status outside {sorted(allowed)}"

    for column in ("time", "start", "stop"):
        if column in df.columns:
            assert (df[column] >= 0).all(), f"{model} has a negative {column}"

    if model in INTERVAL_LAYOUT:
        # Strictly greater, not `>=`. A zero-length risk interval contributes
        # no exposure and cannot carry an event, so it is never a legitimate
        # row -- and `>=` is exactly what let the `tdcm` overflow through:
        # `exp(beta[0] * z)` reaching `inf` collapsed every time to 0.0 and
        # `status = (t <= c)` then called it an observed event at time zero.
        # See `test_tdcm_raises_rather_than_emitting_zero_time_events`.
        assert (
            df["stop"] > df["start"]
        ).all(), f"{model} produced a zero-length or inverted interval"

    if model in ONE_ROW_PER_SUBJECT and "time" in df.columns:
        assert (df["time"] > 0).all(), f"{model} produced an event at time zero"

    if model in ONE_ROW_PER_SUBJECT:
        assert len(df) == n, f"{model} returned {len(df)} rows for n={n}"

    if "id" in df.columns:
        assert df["id"].nunique() <= n, f"{model} invented subjects"

    if model == "thmm":
        assert set(df["state"].unique()) <= {1, 2, 3}
    if model == "cmm":
        assert set(df["from_state"].unique()) <= {1, 2}
        assert set(df["to_state"].unique()) <= {2, 3}
    if model == "recurrent_events":
        assert (df["enum"] >= 1).all()
    if model == "mixture_cure":
        assert set(df["cured"].unique()) <= {0, 1}


@pytest.mark.parametrize("model", sorted(MODEL_STRATEGIES))
def test_output_invariants_hold_for_any_valid_parameters(model: str) -> None:
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(kwargs=strategy, seed=SEED)
    def check(kwargs: Dict[str, Any], seed: int) -> None:
        df = generate(model=model, **kwargs, seed=seed)
        _assert_invariants(model, df, kwargs)

    check()


@pytest.mark.parametrize("model", sorted(MODEL_STRATEGIES))
def test_the_same_seed_gives_the_same_frame(model: str) -> None:
    """The first row of the reproducibility contract, for all twelve models."""
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(kwargs=strategy, seed=SEED)
    def check(kwargs: Dict[str, Any], seed: int) -> None:
        first = generate(model=model, **kwargs, seed=seed)
        second = generate(model=model, **kwargs, seed=seed)
        pd.testing.assert_frame_equal(first, second)

    check()


@pytest.mark.parametrize("model", sorted(MODEL_STRATEGIES))
def test_an_integer_seed_agrees_with_the_generator_it_seeds(model: str) -> None:
    """``seed=7`` and ``seed=default_rng(7)`` are documented as equivalent.

    ``resolve_rng`` passes an ``int`` to ``default_rng`` and returns a
    ``Generator`` unchanged, so the two must draw the same stream. Users rely
    on this to share one generator across several simulators.
    """
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(kwargs=strategy, seed=SEED)
    def check(kwargs: Dict[str, Any], seed: int) -> None:
        from_int = generate(model=model, **kwargs, seed=seed)
        from_generator = generate(
            model=model, **kwargs, seed=np.random.default_rng(seed)
        )
        pd.testing.assert_frame_equal(from_int, from_generator)

    check()


@pytest.mark.parametrize("model", sorted(MODEL_STRATEGIES))
def test_a_non_positive_n_is_always_rejected(model: str) -> None:
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(kwargs=strategy, bad_n=st.integers(min_value=-50, max_value=0))
    def check(kwargs: Dict[str, Any], bad_n: int) -> None:
        with pytest.raises(ValidationError):
            generate(model=model, **{**kwargs, "n": bad_n}, seed=1)

    check()


@pytest.mark.parametrize("model", sorted(MODEL_STRATEGIES))
def test_an_unknown_censoring_model_is_always_rejected(model: str) -> None:
    """Only the models that expose ``model_cens``; the rest have no such knob."""
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(kwargs=strategy, name=st.text(min_size=1, max_size=8))
    def check(kwargs: Dict[str, Any], name: str) -> None:
        if "model_cens" not in kwargs or name in {"uniform", "exponential"}:
            return
        with pytest.raises(ValidationError):
            generate(model=model, **{**kwargs, "model_cens": name}, seed=1)

    check()


@pytest.mark.parametrize(
    "model,argument",
    [
        ("cphm", "covariate_range"),
        ("cphm", "cens_par"),
        ("aft_ln", "sigma"),
        ("aft_weibull", "shape"),
        ("aft_weibull", "scale"),
        ("aft_log_logistic", "shape"),
        ("aft_log_logistic", "scale"),
        ("mixture_cure", "baseline_hazard"),
        ("cmm", "covariate_range"),
        ("thmm", "covariate_range"),
        ("tdcm", "lam"),
        ("recurrent_events", "followup_time"),
    ],
)
def test_a_non_positive_value_is_rejected_where_positivity_is_required(
    model: str, argument: str
) -> None:
    strategy = MODEL_STRATEGIES[model]

    @SETTINGS
    @given(
        kwargs=strategy,
        bad=st.floats(
            min_value=-20.0, max_value=0.0, allow_nan=False, allow_infinity=False
        ),
    )
    def check(kwargs: Dict[str, Any], bad: float) -> None:
        with pytest.raises(ValidationError):
            generate(model=model, **{**kwargs, argument: bad}, seed=1)

    check()


# --------------------------------------------------------------------------
# Regressions found by the properties above
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist,corr,n_par",
    [
        ("weibull", 1.0, 4),
        ("exponential", 1.0, 2),
        ("exponential", -1.0, 2),
    ],
)
def test_tdcm_rejects_the_correlation_endpoints_at_its_own_boundary(
    dist: str, corr: float, n_par: int
) -> None:
    """Two validators disagreed about ``corr``, and the wider one ran first.

    ``validate_gen_tdcm_inputs`` allowed ``0 < corr <= 1`` for Weibull and
    ``-1 <= corr <= 1`` for exponential. The Gaussian copula underneath needs
    strict inequalities -- its covariance is ``[[1, corr], [corr, 1]]``, which
    is singular at ``|corr| = 1`` -- and ``validate_dg_biv_inputs`` enforced
    that. So the endpoints passed the model's own check, failed deeper in, and
    reported "must be a numeric value between -1 and 1": a different range from
    the one the model advertised, from a helper the caller never named. The
    documented range said `(0, 1]` and `[-1, 1]` too.

    The error must now come from the model's boundary and quote the real range.
    """
    with pytest.raises(ValidationError, match="corr"):
        generate(
            model="tdcm",
            n=5,
            dist=dist,
            corr=corr,
            dist_par=[1.0] * n_par,
            model_cens="uniform",
            cens_par=1.0,
            beta=[0.5, 0.3],
            lam=1.0,
            seed=1,
        )


@pytest.mark.parametrize(
    "dist,corr,n_par", [("weibull", 0.99, 4), ("exponential", -0.99, 2)]
)
def test_tdcm_still_accepts_values_just_inside_the_boundary(
    dist: str, corr: float, n_par: int
) -> None:
    """Tightening the bound must not have moved it past what already worked."""
    df = generate(
        model="tdcm",
        n=20,
        dist=dist,
        corr=corr,
        dist_par=[1.0] * n_par,
        model_cens="uniform",
        cens_par=1.0,
        beta=[0.5, 0.3],
        lam=1.0,
        seed=1,
    )
    assert len(df) == 20


def test_tdcm_raises_rather_than_emitting_zero_time_events() -> None:
    """``exp(beta[0] * z)`` leaving float range used to produce a valid-looking frame.

    The covariate is the second margin of the bivariate draw, built as
    ``(-log(1 - u) / a) ** (1 / b)``. A ``b`` well below 1 is an exponent well
    above 1, so the covariate reaches the tens of thousands and
    ``exp(beta[0] * z)`` overflows.

    What came back was not an error. ``t1 = log_term / inf`` is exactly 0.0, and
    ``status = (t <= c)`` then reported an **observed event at time zero** for
    every subject, in a zero-length risk interval. With the sign of ``beta[0]``
    flipped the same expression underflowed to 0.0, making ``t`` infinite and
    every subject censored. Both frames had the right columns, the right dtypes
    and no NaN, so a finiteness check passed them.
    """
    with pytest.raises(ValidationError, match="exp"):
        generate(
            model="tdcm",
            n=2,
            dist="weibull",
            corr=0.5,
            dist_par=[1.0, 1.0, 0.125, 0.125],
            model_cens="uniform",
            cens_par=1.0,
            beta=[1.0, 0.0],
            lam=1.0,
            seed=0,
        )

    with pytest.raises(ValidationError, match="exp"):
        generate(
            model="tdcm",
            n=2,
            dist="weibull",
            corr=0.5,
            dist_par=[1.0, 1.0, 0.125, 0.125],
            model_cens="uniform",
            cens_par=1.0,
            beta=[-1.0, 0.0],
            lam=1.0,
            seed=0,
        )
