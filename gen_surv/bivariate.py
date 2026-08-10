from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

from ._rng import RandomStateLike, resolve_rng
from .validate import validate_dg_biv_inputs

_CLIP_EPS = 1e-10


def sample_bivariate_distribution(
    n: int,
    dist: str,
    corr: float,
    dist_par: Sequence[float],
    seed: RandomStateLike = None,
) -> NDArray[np.float64]:
    """Draw dependent samples with Weibull or exponential marginals.

    Dependence is induced with a Gaussian copula: a pair of correlated standard
    normals is mapped to uniforms through the normal CDF, and those uniforms are
    pushed through the inverse marginal CDFs.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    dist : {"weibull", "exponential"}
        Type of marginal distributions.
    corr : float
        Correlation of the underlying normals, in ``(-1, 1)``. Negative values
        produce negative dependence. Note that this is the correlation on the
        latent normal scale; because the marginals are skewed, the Pearson
        correlation of the returned values is smaller in magnitude, while the
        rank correlation is preserved.
    dist_par : Sequence[float]
        Distribution parameters ``[a1, b1, a2, b2]`` for the Weibull case or
        ``[lambda1, lambda2]`` for the exponential case.
    seed : int or numpy.random.Generator, optional
        Seed or generator for reproducibility.

    Returns
    -------
    NDArray[np.float64]
        Array of shape ``(n, 2)`` with the sampled pairs.

    Examples
    --------
    >>> from gen_surv.bivariate import sample_bivariate_distribution
    >>> sample_bivariate_distribution(
    ...     3,
    ...     "weibull",
    ...     0.3,
    ...     [1.0, 2.0, 1.5, 2.5],
    ...     seed=42,
    ... )  # doctest: +ELLIPSIS
    array([[...], [...], [...]])

    Raises
    ------
    ValidationError
        If ``dist`` is unsupported or ``dist_par`` has an invalid length.
    """

    validate_dg_biv_inputs(n, dist, corr, dist_par)
    rng = resolve_rng(seed)

    # Correlated standard normals, then the probability integral transform.
    # Applying the normal CDF is what makes the marginals exact and keeps the
    # sign of ``corr``. Squaring the normals instead -- as releases up to 1.2.0
    # did -- yields chi-squared marginals and maps both +r and -r onto the same
    # positive dependence, so negative dependence became unreachable.
    cov = [[1.0, corr], [corr, 1.0]]
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n)
    u = np.clip(ndtr(z), _CLIP_EPS, 1 - _CLIP_EPS)

    # Inverse marginal CDFs.
    if dist == "exponential":
        x1 = -np.log(1 - u[:, 0]) / dist_par[0]
        x2 = -np.log(1 - u[:, 1]) / dist_par[1]

    else:  # dist == "weibull"
        a1, b1, a2, b2 = dist_par
        x1 = (-np.log(1 - u[:, 0]) / a1) ** (1 / b1)
        x2 = (-np.log(1 - u[:, 1]) / a2) ** (1 / b2)

    return np.column_stack([x1, x2])
