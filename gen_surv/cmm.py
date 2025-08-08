import pandas as pd
import numpy as np
from gen_surv.validate import validate_gen_cmm_inputs
from gen_surv.censoring import runifcens, rexpocens


def generate_event_times(z1: float, beta: list, rate: list) -> dict:
    """
    Generate event times for a continuous-time multi-state Markov model.

    Parameters:
    - z1 (float): Covariate value
    - beta (list of float): List of 3 beta coefficients
    - rate (list of float): List of 6 transition rate parameters

    Returns:
    - dict: {'t12': float, 't13': float, 't23': float}
    """
    u = np.random.uniform()
    t12 = (-np.log(1 - u) / (rate[0] * np.exp(beta[0] * z1)))**(1 / rate[1])

    u = np.random.uniform()
    t13 = (-np.log(1 - u) / (rate[2] * np.exp(beta[1] * z1)))**(1 / rate[3])

    u = np.random.uniform()
    t23 = (-np.log(1 - u) / (rate[4] * np.exp(beta[2] * z1)))**(1 / rate[5])

    return {"t12": t12, "t13": t13, "t23": t23}

def gen_cmm(n, model_cens, cens_par, beta, covar, rate):
    """
    Generate survival data using a continuous-time Markov model (CMM).

    Parameters:
    - n (int): Number of individuals.
    - model_cens (str): "uniform" or "exponential".
    - cens_par (float): Parameter for censoring.
    - beta (list): Regression coefficients (length 3).
    - covar (float): Covariate range (uniformly sampled from [0, covar]).
    - rate (list): Transition rates (length 6).

    Returns:
    - pd.DataFrame with columns: id, start, stop, status, covariate, transition
    """
    validate_gen_cmm_inputs(n, model_cens, cens_par, beta, covar, rate)

    rfunc = runifcens if model_cens == "uniform" else rexpocens
    rows = []

    for k in range(n):
        z1 = np.random.uniform(0, covar)
        c = rfunc(1, cens_par)[0]
        events = generate_event_times(z1, beta, rate)

        t12, t13, t23 = events["t12"], events["t13"], events["t23"]
        min_event_time = min(t12, t13, c)

        if min_event_time < c:
            if t12 <= t13:
                transition = 1  # 1 -> 2
                rows.append([k + 1, 0, t12, 1, z1, transition])
            else:
                transition = 2  # 1 -> 3
                rows.append([k + 1, 0, t13, 1, z1, transition])
        else:
            # Censored before any event
            rows.append([k + 1, 0, c, 0, z1, np.nan])

    return pd.DataFrame(rows, columns=["id", "start", "stop", "status", "covariate", "transition"])

