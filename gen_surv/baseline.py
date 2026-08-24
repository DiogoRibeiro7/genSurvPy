"""Baseline hazard families.

A survival time is drawn by inverting the cumulative hazard: draw
:math:`E \\sim \\mathrm{Exponential}(1)` and solve :math:`H_0(t) = E / e^{\\eta}`
for :math:`t`. Every generator in this package that samples a continuous time
does exactly that, differing only in which :math:`H_0` it uses.

This module makes that the explicit contract. A baseline hazard is anything
implementing :class:`BaselineHazard`, so a simulator written against the
protocol works with any shape, rather than growing a separate entry point per
family.

All implementations are frozen dataclasses that validate their parameters on
construction, and every method accepts either a scalar or a NumPy array.

Examples
--------
>>> from gen_surv.baseline import WeibullBaseline
>>> baseline = WeibullBaseline(shape=2.0, scale=1.5)
>>> value = baseline.cumulative_hazard(3.0)
>>> round(baseline.inverse_cumulative_hazard(value), 10)
3.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Union, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .validation import ParameterError, ensure_positive, validate_piecewise_params

#: Either a single time or an array of them.
TimeLike = Union[float, NDArray[np.float64]]


@runtime_checkable
class BaselineHazard(Protocol):
    """The interface a baseline hazard must provide to be sampled from.

    Three methods, of which the last two are the ones sampling needs:
    :meth:`cumulative_hazard` to evaluate :math:`H_0(t)`, and
    :meth:`inverse_cumulative_hazard` to solve :math:`H_0(t) = v` for
    :math:`t`. They must be mutual inverses wherever :math:`H_0` is finite and
    strictly increasing.
    """

    def hazard(self, t: TimeLike) -> TimeLike:
        """Instantaneous hazard :math:`h_0(t)`."""
        ...

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        """Integrated hazard :math:`H_0(t) = \\int_0^t h_0(u)\\,du`."""
        ...

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        """Time at which the cumulative hazard reaches ``value``.

        Returns ``inf`` where the cumulative hazard never reaches it, which is a
        real property of some families rather than an error.
        """
        ...


def _as_array(value: TimeLike) -> tuple[NDArray[np.float64], bool]:
    """Return ``value`` as an array, plus whether the input was scalar."""
    array = np.asarray(value, dtype=float)
    return np.atleast_1d(array), array.ndim == 0


def _restore(result: NDArray[np.float64], scalar: bool) -> TimeLike:
    """Return a scalar when the call was made with one."""
    return float(result[0]) if scalar else result


@dataclass(frozen=True)
class ExponentialBaseline:
    """Constant hazard: :math:`h_0(t) = \\lambda`.

    The memoryless case. Waiting times are exponential, so on a multi-event
    process it makes no difference whether the clock runs forward or resets.

    Parameters
    ----------
    rate : float
        The constant hazard, positive.
    """

    rate: float = 1.0

    def __post_init__(self) -> None:
        ensure_positive(self.rate, "rate")

    def hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        return _restore(np.full_like(times, self.rate), scalar)

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        return _restore(self.rate * times, scalar)

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        values, scalar = _as_array(value)
        return _restore(values / self.rate, scalar)


@dataclass(frozen=True)
class WeibullBaseline:
    """Monotone hazard: :math:`H_0(t) = (t/\\sigma)^{\\rho}`.

    Falling for ``shape < 1``, constant at 1, rising above it.

    Parameters
    ----------
    shape : float
        The Weibull shape :math:`\\rho`, positive.
    scale : float
        The Weibull scale :math:`\\sigma`, positive.
    """

    shape: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        ensure_positive(self.shape, "shape")
        ensure_positive(self.scale, "scale")

    def hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        result = (self.shape / self.scale) * (times / self.scale) ** (self.shape - 1.0)
        return _restore(result, scalar)

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        return _restore((times / self.scale) ** self.shape, scalar)

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        values, scalar = _as_array(value)
        return _restore(self.scale * values ** (1.0 / self.shape), scalar)


@dataclass(frozen=True)
class GompertzBaseline:
    """Exponentially changing hazard: :math:`h_0(t) = a e^{bt}`.

    A negative ``shape`` is allowed and gives a declining hazard whose total is
    finite, :math:`a/|b|`. Beyond that total the inverse is ``inf``: the event
    never happens, which is the point of the family rather than a failure.

    Parameters
    ----------
    rate : float
        The hazard at time zero, :math:`a`, positive.
    shape : float
        The exponential rate of change, :math:`b`. Non-zero; negative for a
        declining hazard. Use :class:`ExponentialBaseline` for ``b = 0``.
    """

    rate: float = 1.0
    shape: float = 0.1

    def __post_init__(self) -> None:
        ensure_positive(self.rate, "rate")
        if not np.isfinite(self.shape) or self.shape == 0:
            raise ParameterError(
                "shape",
                self.shape,
                "must be a non-zero finite number; use ExponentialBaseline for a "
                "constant hazard",
            )

    def hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        return _restore(self.rate * np.exp(self.shape * times), scalar)

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        result = self.rate / self.shape * np.expm1(self.shape * times)
        return _restore(result, scalar)

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        values, scalar = _as_array(value)
        inner = 1.0 + self.shape * values / self.rate
        result = np.where(
            inner > 0.0, np.log(np.maximum(inner, 1e-300)) / self.shape, np.inf
        )
        return _restore(np.asarray(result, dtype=float), scalar)

    @property
    def total_hazard(self) -> float:
        """The limit of :math:`H_0(t)`, finite only for a declining hazard."""
        return float("inf") if self.shape > 0 else -self.rate / self.shape


@dataclass(frozen=True)
class LogLogisticBaseline:
    """Unimodal hazard: :math:`H_0(t) = \\log(1 + (t/\\sigma)^{\\rho})`.

    The hazard rises to a peak and then decays, which no other family here
    does. Not a proportional-hazards family in its own right, but a perfectly
    good baseline to scale by a linear predictor.

    Parameters
    ----------
    shape : float
        The shape :math:`\\rho`, positive.
    scale : float
        The scale :math:`\\sigma`, positive.
    """

    shape: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        ensure_positive(self.shape, "shape")
        ensure_positive(self.scale, "scale")

    def hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        ratio = (times / self.scale) ** self.shape
        result = (
            (self.shape / self.scale)
            * (times / self.scale) ** (self.shape - 1.0)
            / (1.0 + ratio)
        )
        return _restore(result, scalar)

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        return _restore(np.log1p((times / self.scale) ** self.shape), scalar)

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        values, scalar = _as_array(value)
        return _restore(self.scale * np.expm1(values) ** (1.0 / self.shape), scalar)


@dataclass(frozen=True)
class PiecewiseConstantBaseline:
    """Constant within intervals, jumping between them.

    There is always one more rate than there are breakpoints: ``k`` breakpoints
    cut the timeline into ``k + 1`` pieces, the last one open-ended.

    Parameters
    ----------
    breakpoints : Sequence[float]
        Strictly increasing positive times at which the hazard changes.
    hazard_rates : Sequence[float]
        One positive rate per interval, ``len(breakpoints) + 1`` of them.
    """

    breakpoints: Sequence[float] = ()
    hazard_rates: Sequence[float] = (1.0,)

    # Derived on construction so evaluation and inversion are a binary search
    # rather than a walk over the intervals.
    _edges: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _rates: NDArray[np.float64] = field(init=False, repr=False, compare=False)
    _at_edges: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        validate_piecewise_params(list(self.breakpoints), list(self.hazard_rates))

        edges = np.asarray(self.breakpoints, dtype=float)
        rates = np.asarray(self.hazard_rates, dtype=float)
        widths = np.diff(np.concatenate(([0.0], edges)))

        object.__setattr__(self, "_edges", edges)
        object.__setattr__(self, "_rates", rates)
        object.__setattr__(
            self,
            "_at_edges",
            np.concatenate(([0.0], np.cumsum(rates[:-1] * widths))),
        )

    def hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        index = np.searchsorted(self._edges, times, side="right")
        return _restore(self._rates[index], scalar)

    def cumulative_hazard(self, t: TimeLike) -> TimeLike:
        times, scalar = _as_array(t)
        index = np.searchsorted(self._edges, times, side="right")
        starts = np.concatenate(([0.0], self._edges))[index]
        result = self._at_edges[index] + self._rates[index] * (times - starts)
        return _restore(result, scalar)

    def inverse_cumulative_hazard(self, value: TimeLike) -> TimeLike:
        values, scalar = _as_array(value)
        index = np.searchsorted(self._at_edges, values, side="right") - 1
        index = np.clip(index, 0, len(self._rates) - 1)
        starts = np.concatenate(([0.0], self._edges))[index]
        consumed = values - self._at_edges[index]
        return _restore(starts + consumed / self._rates[index], scalar)


#: Baselines addressable by name, for callers that take a string.
BASELINES: dict[str, type] = {
    "exponential": ExponentialBaseline,
    "weibull": WeibullBaseline,
    "gompertz": GompertzBaseline,
    "log_logistic": LogLogisticBaseline,
    "piecewise": PiecewiseConstantBaseline,
}
