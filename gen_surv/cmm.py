import pandas as pd
import numpy as np
from gen_surv.validate import validate_gen_cmm_inputs
from gen_surv.censoring import runifcens, rexpocens
from gen_surv.cmm import generate_event_times

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

