import numpy as np

def runifcens(size: int, cens_par: float) -> np.ndarray:
    """
    Generate uniform censoring times.

    Parameters:
    - size (int): Number of samples.
    - cens_par (float): Upper bound for uniform distribution.

    Returns:
    - np.ndarray of censoring times.
    """
    return np.random.uniform(0, cens_par, size)

def rexpocens(size: int, cens_par: float) -> np.ndarray:
    """
    Generate exponential censoring times.

    Parameters:
    - size (int): Number of samples.
    - cens_par (float): Mean of exponential distribution.

    Returns:
    - np.ndarray of censoring times.
    """
    return np.random.exponential(scale=cens_par, size=size)
