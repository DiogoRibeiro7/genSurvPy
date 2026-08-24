"""Recurrent event data generation.

Subjects may experience the same event repeatedly during follow-up. The three
processes here correspond to the models the data is usually analysed with:

``ag``
    Andersen-Gill. The intensity depends on the covariates but not on how many
    events have already happened, and the clock runs forward from entry. A
    non-homogeneous Poisson process.
``pwp_tt``
    Prentice-Williams-Peterson in total time. As Andersen-Gill, but the
    intensity is scaled by a factor specific to the event number, so the risk of
    a second event may differ from the risk of a first. The clock still runs
    forward from entry.
``pwp_gt``
    Prentice-Williams-Peterson in gap time. As ``pwp_tt``, but the clock resets
    after every event, so the baseline hazard is a function of time since the
    previous event rather than time since entry.

All three return counting-process intervals, the canonical layout for
transition data in this package.
"""

from typing import Literal, Sequence, cast

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from ._covariates import generate_covariates, prepare_betas, set_covariate_params
from ._rng import RandomStateLike, resolve_rng
from .baseline import (
    BaselineHazard,
    ExponentialBaseline,
    GompertzBaseline,
    WeibullBaseline,
)
from .censoring import CensoringFunc, rexpocens, runifcens
from .validation import validate_gen_recurrent_events_inputs

Process = Literal["ag", "pwp_tt", "pwp_gt"]
Baseline = Literal["exponential", "weibull", "gompertz"]

_COLUMNS = ["id", "start", "stop", "status", "enum"]

# Default parameters per baseline, used when ``baseline_params`` is omitted.
_BASELINE_DEFAULTS: dict[str, dict[str, float]] = {
    "exponential": {"rate": 1.0},
    "weibull": {"shape": 1.5, "scale": 1.0},
    "gompertz": {"rate": 0.5, "shape": 0.2},
}

_BASELINE_CLASSES: dict[str, type] = {
    "exponential": ExponentialBaseline,
    "weibull": WeibullBaseline,
    "gompertz": GompertzBaseline,
}


def _resolve_baseline(
    baseline: Baseline | BaselineHazard,
    baseline_params: dict[str, float] | None,
) -> BaselineHazard:
    """Turn the ``baseline`` argument into a :class:`BaselineHazard`.

    A name is looked up and constructed with its defaults filled in; an object
    implementing the protocol is used as it stands, which is how a caller
    supplies a family this module does not name.
    """
    if not isinstance(baseline, str):
        return baseline

    params = dict(_BASELINE_DEFAULTS[baseline])
    params.update(baseline_params or {})
    return cast(BaselineHazard, _BASELINE_CLASSES[baseline](**params))


def _stratum_factor(stratum_effects: Sequence[float] | None, enum: int) -> float:
    """Return the intensity multiplier for the ``enum``-th event.

    The last supplied effect applies to every later event, so a two-element
    sequence describes "first event, and all subsequent ones".
    """
    if not stratum_effects:
        return 1.0
    index = min(enum, len(stratum_effects)) - 1
    return float(stratum_effects[index])


def _subject_rows(
    subject: int,
    eta: float,
    end: float,
    process: Process,
    baseline: BaselineHazard,
    stratum_effects: Sequence[float] | None,
    max_events: int | None,
    rng: Generator,
) -> list[tuple[int, float, float, int, int]]:
    """Generate one subject's counting-process intervals.

    Events are drawn by inversion: with intensity ``h0(t) exp(eta) s_k``, the
    cumulative hazard between consecutive events is Exponential(1), so the next
    event solves ``H0(t) = H0(t_prev) + E / (exp(eta) s_k)`` on a forward clock,
    and ``H0(w) = E / (exp(eta) s_k)`` on a clock that resets.
    """
    rows: list[tuple[int, float, float, int, int]] = []
    linear = float(np.exp(eta))
    current = 0.0
    enum = 1

    while True:
        factor = linear * _stratum_factor(stratum_effects, enum)
        draw = float(rng.exponential())
        consumed = draw / factor

        if process == "pwp_gt":
            gap = float(baseline.inverse_cumulative_hazard(consumed))
            candidate = current + gap
        else:
            target = float(baseline.cumulative_hazard(current)) + consumed
            candidate = float(baseline.inverse_cumulative_hazard(target))

        if not np.isfinite(candidate) or candidate >= end:
            break

        rows.append((subject, current, candidate, 1, enum))
        current = candidate

        if max_events is not None and enum >= max_events:
            # Follow-up stops with the capped event: the subject leaves the
            # study rather than remaining at risk with its events suppressed.
            return rows

        enum += 1

    # The remainder of follow-up, during which no further event occurred.
    rows.append((subject, current, end, 0, enum))
    return rows


def gen_recurrent_events(
    n: int,
    process: Process = "ag",
    baseline: Baseline | BaselineHazard = "exponential",
    baseline_params: dict[str, float] | None = None,
    betas: Sequence[float] | None = None,
    n_covariates: int = 2,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: dict[str, float] | None = None,
    stratum_effects: Sequence[float] | None = None,
    max_events: int | None = None,
    followup_time: float = 10.0,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 20.0,
    seed: RandomStateLike = None,
) -> pd.DataFrame:
    """Generate recurrent event data in counting-process form.

    Parameters
    ----------
    n : int
        Number of subjects. Each contributes one row per at-risk interval, so
        the frame is longer than ``n``.
    process : {"ag", "pwp_tt", "pwp_gt"}
        Event process. ``ag`` is Andersen-Gill, whose intensity ignores the
        event history. ``pwp_tt`` and ``pwp_gt`` are Prentice-Williams-Peterson
        in total and gap time, whose intensity is scaled per event number by
        ``stratum_effects``; ``pwp_gt`` additionally resets the clock after each
        event.
    baseline : {"exponential", "weibull", "gompertz"}
        Baseline hazard family. Exponential is constant, Weibull is monotone,
        Gompertz is exponentially increasing or decreasing.
    baseline_params : dict[str, float], optional
        Parameters of the baseline. ``{"rate"}`` for exponential,
        ``{"shape", "scale"}`` for Weibull, ``{"rate", "shape"}`` for Gompertz.
        Defaults are filled in when omitted.
    betas : Sequence[float], optional
        Coefficients acting on the log intensity, one per covariate. Drawn at
        random when omitted, which is convenient for a smoke test and unusable
        for validation.
    n_covariates : int
        Number of covariates when ``betas`` is not supplied.
    covariate_dist : {"normal", "uniform", "binary"}
        Distribution the covariates are drawn from.
    covariate_params : dict[str, float], optional
        Parameters of that distribution. Defaults are filled in when omitted.
    stratum_effects : Sequence[float], optional
        Multiplicative intensity factors by event number, for the two PWP
        processes. The final entry applies to all later events, so
        ``[1.0, 2.0]`` means "first event at the baseline rate, every subsequent
        one at twice that". Supplying it with ``process="ag"`` raises, because
        an Andersen-Gill intensity cannot depend on the event number.
    max_events : int, optional
        Stop following a subject once it has this many events. ``None`` places
        no cap.
    followup_time : float
        Administrative end of follow-up, applied to every subject.
    model_cens : {"uniform", "exponential"}
        Random dropout mechanism, applied on top of ``followup_time``.
    cens_par : float
        Parameter of the dropout distribution: the upper bound for ``uniform``,
        the mean for ``exponential``.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        Counting-process intervals with columns ``["id", "start", "stop",
        "status", "enum", "X0", ..., "Xp"]``. Each row is the interval over
        which a subject was at risk of its ``enum``-th event; ``status`` is 1 if
        that event occurred at ``stop`` and 0 if follow-up ended first. Every
        subject contributes at least one row, and intervals are contiguous
        within a subject.

    Raises
    ------
    ValidationError
        If any parameter is outside its allowed range.

    Examples
    --------
    >>> from gen_surv.recurrent import gen_recurrent_events
    >>> df = gen_recurrent_events(
    ...     n=50,
    ...     process="ag",
    ...     baseline_params={"rate": 0.5},
    ...     betas=[0.4, -0.2],
    ...     followup_time=5.0,
    ...     seed=42,
    ... )
    >>> list(df.columns)
    ['id', 'start', 'stop', 'status', 'enum', 'X0', 'X1']
    """
    validate_gen_recurrent_events_inputs(
        n=n,
        process=process,
        baseline=baseline,
        baseline_params=baseline_params,
        n_covariates=n_covariates,
        stratum_effects=stratum_effects,
        max_events=max_events,
        followup_time=followup_time,
        model_cens=model_cens,
        cens_par=cens_par,
    )

    rng = resolve_rng(seed)

    resolved_baseline = _resolve_baseline(baseline, baseline_params)

    covariate_params = set_covariate_params(covariate_dist, covariate_params)
    coefficients, n_covariates = prepare_betas(betas, n_covariates, rng, name="betas")
    covariates: NDArray[np.float64] = generate_covariates(
        n, n_covariates, covariate_dist, covariate_params, rng
    )
    eta = covariates @ coefficients

    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens
    dropout = rfunc(n, cens_par, rng)
    ends = np.minimum(dropout, followup_time)

    # The event history of one subject depends on its own previous draws, so
    # this loop is over subjects rather than over a vectorised array.
    records: list[tuple[int, float, float, int, int]] = []
    for subject in range(n):
        records.extend(
            _subject_rows(
                subject=subject,
                eta=float(eta[subject]),
                end=float(ends[subject]),
                process=process,
                baseline=resolved_baseline,
                stratum_effects=stratum_effects,
                max_events=max_events,
                rng=rng,
            )
        )

    data = pd.DataFrame(records, columns=_COLUMNS)
    data = data.astype(
        {
            "id": "int64",
            "start": "float64",
            "stop": "float64",
            "status": "int64",
            "enum": "int64",
        }
    )

    for j in range(n_covariates):
        data[f"X{j}"] = covariates[data["id"].to_numpy(), j]

    return data.reset_index(drop=True)
