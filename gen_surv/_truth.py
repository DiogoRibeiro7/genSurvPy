"""A scoped sink for the ground truth a generator computes but does not return.

Generators build quantities no real dataset could contain — the coefficients
actually used, the latent event time before censoring intervened, which
subjects are cured — and then discard all but the frame.

Rather than changing twelve signatures or returning tuples the ordinary caller
does not want, a generator calls :func:`record` at the point where those values
exist. Outside a :func:`capture` block that is a no-op, so ``gen_*`` behaves
exactly as before; inside one, the values land in the dictionary
:func:`gen_surv.simulate` returns.

The sink is a :class:`~contextvars.ContextVar`, so it is per-thread and
per-task and cannot leak between concurrent simulations.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

_sink: ContextVar[dict[str, Any] | None] = ContextVar(
    "gen_surv_truth_sink", default=None
)


def record(**values: Any) -> None:
    """Record ground truth, if anything is capturing it.

    Call this where the values exist, immediately before returning the frame.
    Outside a :func:`capture` block it does nothing at all.

    Parameters
    ----------
    **values
        Named ground-truth quantities. Use the shared vocabulary where it
        applies — ``beta``/``betas``, ``covariates``, ``linear_predictor``,
        ``event_time``, ``censoring_time`` — so results are comparable across
        models, and model-specific names otherwise.
    """
    sink = _sink.get()
    if sink is not None:
        sink.update(values)


def current() -> dict[str, Any] | None:
    """Return the active sink, or ``None`` when nothing is capturing.

    A generator that delegates to another one uses this to translate what the
    inner call recorded into its own vocabulary.
    """
    return _sink.get()


@contextmanager
def capture() -> Iterator[dict[str, Any]]:
    """Collect everything :func:`record` reports inside this block.

    Examples
    --------
    >>> from gen_surv._truth import capture
    >>> from gen_surv import gen_cphm
    >>> with capture() as truth:
    ...     frame = gen_cphm(n=10, model_cens="uniform", cens_par=1.0,
    ...                      beta=0.5, covariate_range=2.0, seed=1)
    >>> sorted(truth)
    ['beta', 'censoring_time', 'covariates', 'event_time', 'linear_predictor']
    """
    sink: dict[str, Any] = {}
    token = _sink.set(sink)
    try:
        yield sink
    finally:
        _sink.reset(token)
