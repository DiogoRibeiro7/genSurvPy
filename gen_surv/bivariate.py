from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .validate import validate_dg_biv_inputs

_CHI2_SCALE = 0.5
_CLIP_EPS = 1e-10


def sample_bivariate_distribution(
    n: int, dist: str, corr: float, dist_par: Sequence[float]
) -> NDArray[np.float64]:
    """Draw correlated samples from Weibull or exponential marginals.

    Parameters
    ----------
    n : int
        Number of samples to generate.
    dist : {"weibull", "exponential"}
        Type of marginal distributions.
    corr : float
        Correlation coefficient.
    dist_par : Sequence[float]
        Distribution parameters ``[a1, b1, a2, b2]`` for the Weibull case or
        ``[lambda1, lambda2]`` for the exponential case.

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
    ... )  # doctest: +ELLIPSIS
    array([[...], [...], [...]])

    Raises
    ------
    ValidationError
        If ``dist`` is unsupported or ``dist_par`` has an invalid length.
    """

    validate_dg_biv_inputs(n, dist, corr, dist_par)

    # Step 1: Generate correlated standard normals using Cholesky
    mean = [0, 0]
    cov = [[1, corr], [corr, 1]]
    z = np.random.multivariate_normal(mean, cov, size=n)
    u = 1 - np.exp(
        -_CHI2_SCALE * z**2
    )  # transform normals to uniform via chi-squared approx
    u = np.clip(u, _CLIP_EPS, 1 - _CLIP_EPS)  # avoid infs in tails

    # Step 2: Transform to marginals
    if dist == "exponential":
        x1 = -np.log(1 - u[:, 0]) / dist_par[0]
        x2 = -np.log(1 - u[:, 1]) / dist_par[1]

    else:  # dist == "weibull"
        a1, b1, a2, b2 = dist_par
        x1 = (-np.log(1 - u[:, 0]) / a1) ** (1 / b1)
        x2 = (-np.log(1 - u[:, 1]) / a2) ** (1 / b2)

    return np.column_stack([x1, x2])
