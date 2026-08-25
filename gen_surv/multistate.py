"""A general multistate engine.

A subject moves through a graph of states. Each edge carries its own baseline
hazard and its own coefficients, so the intensity of the ``i -> j`` transition
is

.. math::

    \\alpha_{ij}(t \\mid X) = h_{0,ij}(t)\\exp(X^\\top\\beta_{ij}).

Two clocks are supported, and the choice is what separates a Markov process
from a semi-Markov one:

``clock="forward"``
    The hazard is a function of time since entry to the study. The process is
    Markov: where a subject has been does not matter, only where it is and how
    long the study has run.
``clock="reset"``
    The hazard restarts at each entry to a state, so it is a function of time
    in the current state. The process is semi-Markov.

With an exponential baseline the two coincide, because a constant hazard is
memoryless.

Both canonical layouts are available. ``layout="intervals"`` gives
counting-process rows -- one per transition a subject was at risk of, over the
interval it was at risk -- and ``layout="panel"`` gives one row per observation
of the subject's state. See :doc:`the output schemas page </getting-started/schemas>`.

Examples
--------
An illness-death process with Weibull sojourns:

>>> from gen_surv import Transition, WeibullBaseline, gen_multistate
>>> transitions = [
...     Transition(1, 2, WeibullBaseline(shape=1.0, scale=3.0), [0.3]),
...     Transition(1, 3, WeibullBaseline(shape=1.0, scale=5.0), [0.1]),
...     Transition(2, 3, WeibullBaseline(shape=1.2, scale=2.0), [0.2]),
... ]
>>> frame = gen_multistate(n=100, transitions=transitions, clock="reset", seed=1)
>>> list(frame.columns)
['id', 'start', 'stop', 'from_state', 'to_state', 'status', 'X0']
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from ._covariates import generate_covariates, set_covariate_params
from ._rng import RandomStateLike, resolve_rng
from ._truth import record
from .baseline import BaselineHazard
from .censoring import CensoringFunc, rexpocens, runifcens
from .validation import (
    ParameterError,
    ensure_in_choices,
    ensure_numeric_sequence,
    ensure_positive,
    ensure_positive_int,
)

Clock = Literal["forward", "reset"]
Layout = Literal["intervals", "panel"]

INTERVAL_COLUMNS = ["id", "start", "stop", "from_state", "to_state", "status"]
PANEL_COLUMNS = ["id", "time", "state"]

#: A subject cannot make more transitions than this. Only reachable with a
#: cyclic graph and a very high intensity; it turns a runaway into an error.
_MAX_TRANSITIONS = 10_000


@dataclass(frozen=True)
class Transition:
    """One edge of the transition graph.

    Parameters
    ----------
    origin : int
        The state a subject moves from.
    destination : int
        The state it moves to. Must differ from ``origin``.
    baseline : BaselineHazard
        The baseline hazard for this transition. Any object implementing the
        protocol works, so the shape is a parameter rather than a fork in the
        code.
    coefficients : Sequence[float]
        One coefficient per covariate, acting on the log intensity. Empty means
        the transition does not depend on the covariates.
    """

    origin: int
    destination: int
    baseline: BaselineHazard
    coefficients: Sequence[float] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("origin", "destination"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ParameterError(name, value, "must be an integer state label")
        if self.origin == self.destination:
            raise ParameterError(
                "destination",
                self.destination,
                f"must differ from origin ({self.origin}); a transition to the "
                "same state has no meaning",
            )
        required = ("cumulative_hazard", "inverse_cumulative_hazard")
        if not all(callable(getattr(self.baseline, name, None)) for name in required):
            raise ParameterError(
                "baseline",
                self.baseline,
                "must implement BaselineHazard, with cumulative_hazard and "
                "inverse_cumulative_hazard",
            )
        ensure_numeric_sequence(list(self.coefficients), "coefficients")
        object.__setattr__(
            self, "coefficients", tuple(float(c) for c in self.coefficients)
        )


def _validate_graph(transitions: Sequence[Transition], initial_state: int) -> int:
    """Check the graph is usable and return the number of covariates it needs."""
    if not transitions:
        raise ParameterError("transitions", transitions, "must not be empty")

    seen: set[tuple[int, int]] = set()
    for transition in transitions:
        edge = (transition.origin, transition.destination)
        if edge in seen:
            raise ParameterError(
                "transitions",
                edge,
                "is listed twice; each origin-destination pair may appear once",
            )
        seen.add(edge)

    widths = {len(t.coefficients) for t in transitions}
    if len(widths) > 1:
        raise ParameterError(
            "transitions",
            sorted(widths),
            "must all carry the same number of coefficients, one per covariate",
        )

    origins = {t.origin for t in transitions}
    if initial_state not in origins:
        raise ParameterError(
            "initial_state",
            initial_state,
            f"has no outgoing transition, so no subject can ever leave it; "
            f"states with transitions are {sorted(origins)}",
        )

    return widths.pop()


def _subject_path(
    eta: NDArray[np.float64],
    end: float,
    outgoing: dict[int, list[tuple[int, Transition]]],
    initial_state: int,
    clock: Clock,
    rng: Generator,
    latent: dict[int, float],
) -> list[tuple[int, float, float, int | None]]:
    """Walk one subject through the graph.

    Returns one entry per state occupancy: ``(state, entry, exit, destination)``
    where ``destination`` is ``None`` when follow-up ended in that state.

    ``latent`` collects the first candidate time drawn for each transition,
    including the ones that lost the race. Those are exactly the quantities a
    real dataset could not contain, so they are worth keeping for
    :func:`gen_surv.simulate`.
    """
    occupancies: list[tuple[int, float, float, int | None]] = []
    state = initial_state
    now = 0.0
    entered = 0.0

    for _ in range(_MAX_TRANSITIONS):
        edges = outgoing.get(state)
        if not edges:  # absorbing
            return occupancies

        best_time = float("inf")
        best_destination: int | None = None
        for index, transition in edges:
            # The cumulative intensity between transitions is Exponential(1).
            consumed = float(rng.exponential()) / float(np.exp(eta[index]))
            if clock == "reset":
                gap = float(transition.baseline.inverse_cumulative_hazard(consumed))
                candidate = entered + gap
            else:
                target = float(transition.baseline.cumulative_hazard(now)) + consumed
                candidate = float(transition.baseline.inverse_cumulative_hazard(target))
            latent.setdefault(index, candidate)
            if candidate < best_time:
                best_time = candidate
                best_destination = transition.destination

        # A tie between the transition and the end of follow-up goes to the
        # transition, matching the R implementation's `c < min(...)`.
        if not np.isfinite(best_time) or best_time > end:
            occupancies.append((state, entered, end, None))
            return occupancies

        occupancies.append((state, entered, best_time, best_destination))
        assert best_destination is not None
        state = best_destination
        now = best_time
        entered = best_time

    raise ParameterError(
        "transitions",
        len(occupancies),
        f"produced more than {_MAX_TRANSITIONS} transitions for one subject; "
        "check for a cycle with a very high intensity",
    )


def gen_multistate(
    n: int,
    transitions: Sequence[Transition],
    clock: Clock = "forward",
    initial_state: int = 1,
    covariate_dist: Literal["normal", "uniform", "binary"] = "normal",
    covariate_params: dict[str, float] | None = None,
    model_cens: Literal["uniform", "exponential"] = "uniform",
    cens_par: float = 5.0,
    max_time: float | None = None,
    layout: Layout = "intervals",
    seed: RandomStateLike = None,
) -> pd.DataFrame:
    """Simulate a multistate process over an arbitrary transition graph.

    Parameters
    ----------
    n : int
        Number of subjects. Each contributes several rows, so the frame is
        longer than ``n``.
    transitions : Sequence[Transition]
        The graph. Every edge carries its own baseline hazard and coefficients.
        A state with no outgoing transition is absorbing.
    clock : {"forward", "reset"}
        ``"forward"`` measures the hazard from entry to the study, giving a
        Markov process; ``"reset"`` measures it from entry to the current
        state, giving a semi-Markov one. They coincide for an exponential
        baseline.
    initial_state : int
        The state every subject starts in, at time zero.
    covariate_dist : {"normal", "uniform", "binary"}
        Distribution the covariates are drawn from.
    covariate_params : dict[str, float], optional
        Parameters of that distribution; defaults are filled in.
    model_cens : {"uniform", "exponential"}
        Random censoring mechanism.
    cens_par : float
        Parameter of the censoring distribution.
    max_time : float, optional
        Administrative end of follow-up, applied on top of random censoring.
    layout : {"intervals", "panel"}
        ``"intervals"`` returns counting-process rows: one per transition a
        subject was at risk of, over the interval it was at risk, with
        ``status`` marking the one that occurred. ``"panel"`` returns one row
        per observation of the subject's state.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        For ``layout="intervals"``: ``["id", "start", "stop", "from_state",
        "to_state", "status", "X0", ...]``. For ``layout="panel"``:
        ``["id", "time", "state", "X0", ...]``.

    Raises
    ------
    ValidationError
        If the graph is malformed or any parameter is out of range.
    """
    ensure_positive_int(n, "n")
    ensure_in_choices(clock, "clock", ("forward", "reset"))
    ensure_in_choices(layout, "layout", ("intervals", "panel"))
    ensure_in_choices(model_cens, "model_cens", ("uniform", "exponential"))
    ensure_positive(cens_par, "cens_par")
    if max_time is not None:
        ensure_positive(max_time, "max_time")

    n_covariates = _validate_graph(transitions, initial_state)

    rng = resolve_rng(seed)
    covariate_params = set_covariate_params(covariate_dist, covariate_params)
    covariates = (
        generate_covariates(n, n_covariates, covariate_dist, covariate_params, rng)
        if n_covariates
        else np.zeros((n, 0))
    )

    coefficients = np.array(
        [list(t.coefficients) for t in transitions], dtype=float
    ).reshape(len(transitions), n_covariates)
    # One linear predictor per subject per transition.
    eta = covariates @ coefficients.T

    rfunc: CensoringFunc = runifcens if model_cens == "uniform" else rexpocens
    dropout = rfunc(n, cens_par, rng)
    ends = dropout if max_time is None else np.minimum(dropout, max_time)

    outgoing: dict[int, list[tuple[int, Transition]]] = {}
    for index, transition in enumerate(transitions):
        outgoing.setdefault(transition.origin, []).append((index, transition))

    interval_rows: list[tuple[Any, ...]] = []
    panel_rows: list[tuple[Any, ...]] = []
    # First candidate time per transition per subject; NaN where a subject was
    # never at risk of that transition.
    latent_times = np.full((n, len(transitions)), np.nan)

    for subject in range(n):
        latent: dict[int, float] = {}
        occupancies = _subject_path(
            eta=eta[subject],
            end=float(ends[subject]),
            outgoing=outgoing,
            initial_state=initial_state,
            clock=clock,
            rng=rng,
            latent=latent,
        )
        for index, value in latent.items():
            latent_times[subject, index] = value

        if layout == "intervals":
            for state, entry, exit_time, destination in occupancies:
                for _, transition in outgoing[state]:
                    interval_rows.append(
                        (
                            subject,
                            entry,
                            exit_time,
                            state,
                            transition.destination,
                            int(transition.destination == destination),
                        )
                    )
        else:
            panel_rows.append((subject, 0.0, initial_state))
            for state, _entry, exit_time, destination in occupancies:
                # A transition is observed in its destination; the end of
                # follow-up is observed in the state still occupied.
                panel_rows.append(
                    (
                        subject,
                        exit_time,
                        destination if destination is not None else state,
                    )
                )

    rows = interval_rows if layout == "intervals" else panel_rows
    columns = INTERVAL_COLUMNS if layout == "intervals" else PANEL_COLUMNS
    data = pd.DataFrame(rows, columns=columns)

    dtypes: dict[str, str] = {"id": "int64"}
    if layout == "intervals":
        dtypes.update(
            {
                "start": "float64",
                "stop": "float64",
                "from_state": "int64",
                "to_state": "int64",
                "status": "int64",
            }
        )
    else:
        dtypes.update({"time": "float64", "state": "int64"})
    data = data.astype(dtypes)

    for j in range(n_covariates):
        data[f"X{j}"] = covariates[data["id"].to_numpy(), j]

    record(
        transitions=tuple(transitions),
        clock=clock,
        covariates=covariates,
        linear_predictor=eta,
        censoring_time=dropout,
        followup_end=ends,
        latent_times={
            (t.origin, t.destination): latent_times[:, i]
            for i, t in enumerate(transitions)
        },
    )

    return data.reset_index(drop=True)
