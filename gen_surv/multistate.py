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


def _walk_cohort(
    eta: NDArray[np.float64],
    ends: NDArray[np.float64],
    outgoing: dict[int, list[tuple[int, Transition]]],
    initial_state: int,
    clock: Clock,
    rng: Generator,
    n_transitions: int,
) -> tuple[
    list[
        tuple[
            NDArray[np.int64],
            int,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
        ]
    ],
    NDArray[np.float64],
]:
    """Advance every subject through the graph, a wave at a time.

    Subjects occupying the same state are advanced together, so the draws and
    the inversions are array operations rather than one call per subject. The
    number of waves is the longest path any subject takes, which is small even
    for a cyclic graph, where a per-subject loop costs one Python iteration per
    subject per step.

    Returns the occupancies -- ``(subjects, state, entry, exit, destination)``
    per wave, with ``-1`` marking follow-up that ended in that state -- and the
    first candidate time drawn for each subject and transition.
    """
    n = len(ends)
    state = np.full(n, initial_state, dtype=np.int64)
    entered = np.zeros(n, dtype=float)
    now = np.zeros(n, dtype=float)
    active = np.ones(n, dtype=bool)

    # Absorbing states end follow-up immediately.
    for label in np.unique(state):
        if int(label) not in outgoing:
            active &= state != label

    latent = np.full((n, n_transitions), np.nan)
    occupancies: list[
        tuple[
            NDArray[np.int64],
            int,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.int64],
        ]
    ] = []

    for _ in range(_MAX_TRANSITIONS):
        if not active.any():
            return occupancies, latent

        for label in np.unique(state[active]):
            here = np.flatnonzero(active & (state == label))
            edges = outgoing.get(int(label))
            if not edges:  # pragma: no cover - filtered above
                continue

            candidates = np.empty((len(edges), len(here)))
            for row, (index, transition) in enumerate(edges):
                # The cumulative intensity between transitions is Exponential(1).
                consumed = rng.exponential(size=len(here)) / np.exp(eta[here, index])
                if clock == "reset":
                    gaps = np.asarray(
                        transition.baseline.inverse_cumulative_hazard(consumed)
                    )
                    candidates[row] = entered[here] + gaps
                else:
                    target = (
                        np.asarray(transition.baseline.cumulative_hazard(now[here]))
                        + consumed
                    )
                    candidates[row] = np.asarray(
                        transition.baseline.inverse_cumulative_hazard(target)
                    )
                # Only the first time a subject faces this edge.
                unseen = np.isnan(latent[here, index])
                latent[here[unseen], index] = candidates[row][unseen]

            winner = np.argmin(candidates, axis=0)
            best = candidates[winner, np.arange(len(here))]

            # A tie between the transition and the end of follow-up goes to the
            # transition, matching the R implementation's `c < min(...)`.
            stops = ~np.isfinite(best) | (best > ends[here])
            destinations = np.array(
                [edges[w][1].destination for w in winner], dtype=np.int64
            )
            destinations[stops] = -1
            exits = np.where(stops, ends[here], best)

            occupancies.append(
                (here, int(label), entered[here].copy(), exits, destinations)
            )

            moved = here[~stops]
            active[here[stops]] = False
            if moved.size:
                state[moved] = destinations[~stops]
                now[moved] = exits[~stops]
                entered[moved] = exits[~stops]
                # Reaching a state with no way out ends follow-up.
                for label_out in np.unique(state[moved]):
                    if int(label_out) not in outgoing:
                        active[moved[state[moved] == label_out]] = False

    raise ParameterError(
        "transitions",
        _MAX_TRANSITIONS,
        f"produced more than {_MAX_TRANSITIONS} transitions for a subject; "
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

    occupancies, latent_times = _walk_cohort(
        eta=eta,
        ends=ends,
        outgoing=outgoing,
        initial_state=initial_state,
        clock=clock,
        rng=rng,
        n_transitions=len(transitions),
    )

    # Columns are accumulated as arrays and concatenated once. Building the
    # frame from Python tuples costs more than the sampling does.
    chunks: dict[str, list[NDArray[Any]]] = {}

    def add(**columns: NDArray[Any]) -> None:
        for key, values in columns.items():
            chunks.setdefault(key, []).append(values)

    if layout == "intervals":
        for subjects, state, entry, exit_time, destination in occupancies:
            width = len(subjects)
            for _, transition in outgoing[state]:
                add(
                    id=subjects,
                    start=entry,
                    stop=exit_time,
                    from_state=np.full(width, state, dtype=np.int64),
                    to_state=np.full(width, transition.destination, dtype=np.int64),
                    status=(destination == transition.destination).astype(np.int64),
                )
    else:
        add(
            id=np.arange(n, dtype=np.int64),
            time=np.zeros(n),
            state=np.full(n, initial_state, dtype=np.int64),
        )
        for subjects, state, _entry, exit_time, destination in occupancies:
            add(
                id=subjects,
                time=exit_time,
                # A transition is observed in its destination; the end of
                # follow-up is observed in the state still occupied.
                state=np.where(destination == -1, state, destination).astype(np.int64),
            )

    columns = INTERVAL_COLUMNS if layout == "intervals" else PANEL_COLUMNS
    data = pd.DataFrame({name: np.concatenate(chunks[name]) for name in columns})

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

    # Waves put every subject's first occupancy before anyone's second, so the
    # frame is sorted back into per-subject order.
    order = ["id", "start"] if layout == "intervals" else ["id", "time"]
    if layout == "intervals":
        order.append("to_state")
    data = data.sort_values(order, kind="stable").reset_index(drop=True)

    return data
