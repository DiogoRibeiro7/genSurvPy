from typing import Protocol

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray


class CensoringFunc(Protocol):
    """Protocol for censoring time generators."""

    def __call__(
        self, size: int, cens_par: float, rng: Generator | None = None
    ) -> NDArray[np.float64]:
        """Generate ``size`` censoring times given ``cens_par``."""
        ...


class CensoringModel(Protocol):
    """Protocol for class-based censoring generators."""

    def __call__(self, size: int) -> NDArray[np.float64]:
        """Generate ``size`` censoring times."""
        ...


def runifcens(
    size: int, cens_par: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """
    Generate uniform censoring times.

    Parameters:
    - size (int): Number of samples.
    - cens_par (float): Upper bound for uniform distribution.

    Returns:
    - NDArray of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.uniform(0, cens_par, size)


def rexpocens(
    size: int, cens_par: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """
    Generate exponential censoring times.

    Parameters:
    - size (int): Number of samples.
    - cens_par (float): Mean of exponential distribution.

    Returns:
    - NDArray of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.exponential(scale=cens_par, size=size)


def rweibcens(
    size: int, scale: float, shape: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate Weibull-distributed censoring times."""
    r = default_rng() if rng is None else rng
    return r.weibull(shape, size) * scale


def rlognormcens(
    size: int, mean: float, sigma: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate log-normal-distributed censoring times."""
    r = default_rng() if rng is None else rng
    return r.lognormal(mean, sigma, size)


def rgammacens(
    size: int, shape: float, scale: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate Gamma-distributed censoring times."""
    r = default_rng() if rng is None else rng
    return r.gamma(shape, scale, size)


class WeibullCensoring:
    """Class-based generator for Weibull censoring times."""

    def __init__(self, scale: float, shape: float) -> None:
        self.scale = scale
        self.shape = shape

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a Weibull distribution."""
        r = default_rng() if rng is None else rng
        return r.weibull(self.shape, size) * self.scale


class LogNormalCensoring:
    """Class-based generator for log-normal censoring times."""

    def __init__(self, mean: float, sigma: float) -> None:
        self.mean = mean
        self.sigma = sigma

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a log-normal distribution."""
        r = default_rng() if rng is None else rng
        return r.lognormal(self.mean, self.sigma, size)


class GammaCensoring:
    """Class-based generator for Gamma censoring times."""

    def __init__(self, shape: float, scale: float) -> None:
        self.shape = shape
        self.scale = scale

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a Gamma distribution."""
        r = default_rng() if rng is None else rng
        return r.gamma(self.shape, self.scale, size)
