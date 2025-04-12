import numpy as np

def sample_bivariate_distribution(n, dist, corr, dist_par):
    """
    Generate samples from a bivariate distribution with specified correlation.

    Parameters:
    - n (int): Number of samples
    - dist (str): 'weibull' or 'exponential'
    - corr (float): Correlation coefficient between [-1, 1]
    - dist_par (list): Parameters for the marginals

    Returns:
    - np.ndarray of shape (n, 2)
    """
    if dist not in {"weibull", "exponential"}:
        raise ValueError("Only 'weibull' and 'exponential' distributions are supported.")

    # Step 1: Generate correlated standard normals using Cholesky
    mean = [0, 0]
    cov = [[1, corr], [corr, 1]]
    z = np.random.multivariate_normal(mean, cov, size=n)
    u = 1 - np.exp(-0.5 * z**2)  # transform normals to uniform via chi-squared approx
    u = np.clip(u, 1e-10, 1 - 1e-10)  # avoid infs in tails

    # Step 2: Transform to marginals
    if dist == "exponential":
        if len(dist_par) != 2:
            raise ValueError("Exponential distribution requires 2 positive rate parameters.")
        x1 = -np.log(1 - u[:, 0]) / dist_par[0]
        x2 = -np.log(1 - u[:, 1]) / dist_par[1]

    elif dist == "weibull":
        if len(dist_par) != 4:
            raise ValueError("Weibull distribution requires 4 positive parameters [a1, b1, a2, b2].")
        a1, b1, a2, b2 = dist_par
        x1 = (-np.log(1 - u[:, 0]) / a1) ** (1 / b1)
        x2 = (-np.log(1 - u[:, 1]) / a2) ** (1 / b2)

    return np.column_stack([x1, x2])
