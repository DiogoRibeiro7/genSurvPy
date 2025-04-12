import numpy as np
import pandas as pd
from gen_surv.validate import validate_gen_cphm_inputs
from gen_surv.censoring import runifcens, rexpocens

def generate_cphm_data(n, rfunc, cens_par, beta, covariate_range):
    """
    Generate data from a Cox Proportional Hazards Model (CPHM).

    Parameters:
    - n (int): Number of samples to generate.
    - rfunc (callable): Function to generate censoring times, must accept (size, cens_par).
    - cens_par (float): Parameter passed to the censoring function.
    - beta (float): Coefficient for the covariate.
    - covar (float): Range for the covariate (uniformly sampled from [0, covar]).

    Returns:
    - np.ndarray: Array with shape (n, 3): [time, status, covariate]
    """
    data = np.zeros((n, 3))

    for k in range(n):
        z = np.random.uniform(0, covariate_range)
        c = rfunc(1, cens_par)[0]
        x = np.random.exponential(scale=1 / np.exp(beta * z))

        time = min(x, c)
        status = int(x <= c)

        data[k, :] = [time, status, z]

    return data


def gen_cphm(n: int, model_cens: str, cens_par: float, beta: float, covar: float) -> pd.DataFrame:
    """
    Convenience wrapper to generate CPHM survival data.

    Parameters:
    - n (int): Number of observations.
    - model_cens (str): "uniform" or "exponential".
    - cens_par (float): Parameter for the censoring model.
    - beta (float): Coefficient for the covariate.
    - covar (float): Covariate range (uniform between 0 and covar).

    Returns:
    - pd.DataFrame: Columns are ["time", "status", "covariate"]
    """
    validate_gen_cphm_inputs(n, model_cens, cens_par, covar)

    rfunc = {
        "uniform": runifcens,
        "exponential": rexpocens
    }[model_cens]

    data = generate_cphm_data(n, rfunc, cens_par, beta, covar)

    return pd.DataFrame(data, columns=["time", "status", "covariate"])
