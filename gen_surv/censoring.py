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
    """Generate uniform censoring times.

    Parameters
    ----------
    size : int
        Number of samples.
    cens_par : float
        Upper bound for the uniform distribution.
    rng : Generator, optional
        Random number generator to use. If ``None``, a default generator is
        created.

    Returns
    -------
    NDArray[np.float64]
        Array of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.uniform(0, cens_par, size)


def rexpocens(
    size: int, cens_par: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate exponential censoring times.

    Parameters
    ----------
    size : int
        Number of samples.
    cens_par : float
        Mean of the exponential distribution.
    rng : Generator, optional
        Random number generator to use. If ``None``, a default generator is
        created.

    Returns
    -------
    NDArray[np.float64]
        Array of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.exponential(scale=cens_par, size=size)


def rweibcens(
    size: int, scale: float, shape: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate Weibull-distributed censoring times.

    Parameters
    ----------
    size : int
        Number of samples.
    scale : float
        Scale parameter of the Weibull distribution.
    shape : float
        Shape parameter of the Weibull distribution.
    rng : Generator, optional
        Random number generator to use. If ``None``, a default generator is
        created.

    Returns
    -------
    NDArray[np.float64]
        Array of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.weibull(shape, size) * scale


def rlognormcens(
    size: int, mean: float, sigma: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate log-normal-distributed censoring times.

    Parameters
    ----------
    size : int
        Number of samples.
    mean : float
        Mean of the underlying normal distribution.
    sigma : float
        Standard deviation of the underlying normal distribution.
    rng : Generator, optional
        Random number generator to use. If ``None``, a default generator is
        created.

    Returns
    -------
    NDArray[np.float64]
        Array of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.lognormal(mean, sigma, size)


def rgammacens(
    size: int, shape: float, scale: float, rng: Generator | None = None
) -> NDArray[np.float64]:
    """Generate Gamma-distributed censoring times.

    Parameters
    ----------
    size : int
        Number of samples.
    shape : float
        Shape parameter of the Gamma distribution.
    scale : float
        Scale parameter of the Gamma distribution.
    rng : Generator, optional
        Random number generator to use. If ``None``, a default generator is
        created.

    Returns
    -------
    NDArray[np.float64]
        Array of censoring times.
    """
    r = default_rng() if rng is None else rng
    return r.gamma(shape, scale, size)


class WeibullCensoring:
    """Class-based generator for Weibull censoring times."""

    def __init__(self, scale: float, shape: float) -> None:
        """Store Weibull scale and shape parameters.

        Parameters
        ----------
        scale : float
            Scale parameter of the Weibull distribution.
        shape : float
            Shape parameter of the Weibull distribution.
        """
        self.scale = scale
        self.shape = shape

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a Weibull distribution.

        Parameters
        ----------
        size : int
            Number of samples.
        rng : Generator, optional
            Random number generator to use. If ``None``, a default generator is
            created.

        Returns
        -------
        NDArray[np.float64]
            Array of censoring times.
        """
        r = default_rng() if rng is None else rng
        return r.weibull(self.shape, size) * self.scale


class LogNormalCensoring:
    """Class-based generator for log-normal censoring times."""

    def __init__(self, mean: float, sigma: float) -> None:
        """Store log-normal parameters.

        Parameters
        ----------
        mean : float
            Mean of the underlying normal distribution.
        sigma : float
            Standard deviation of the underlying normal distribution.
        """
        self.mean = mean
        self.sigma = sigma

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a log-normal distribution.

        Parameters
        ----------
        size : int
            Number of samples.
        rng : Generator, optional
            Random number generator to use. If ``None``, a default generator is
            created.

        Returns
        -------
        NDArray[np.float64]
            Array of censoring times.
        """
        r = default_rng() if rng is None else rng
        return r.lognormal(self.mean, self.sigma, size)


class GammaCensoring:
    """Class-based generator for Gamma censoring times."""

    def __init__(self, shape: float, scale: float) -> None:
        """Store Gamma distribution parameters.

        Parameters
        ----------
        shape : float
            Shape parameter of the Gamma distribution.
        scale : float
            Scale parameter of the Gamma distribution.
        """
        self.shape = shape
        self.scale = scale

    def __call__(self, size: int, rng: Generator | None = None) -> NDArray[np.float64]:
        """Generate ``size`` censoring times from a Gamma distribution.

        Parameters
        ----------
        size : int
            Number of samples.
        rng : Generator, optional
            Random number generator to use. If ``None``, a default generator is
            created.

        Returns
        -------
        NDArray[np.float64]
            Array of censoring times.
        """
        r = default_rng() if rng is None else rng
        return r.gamma(self.shape, self.scale, size)
