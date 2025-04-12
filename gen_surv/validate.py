def validate_gen_cphm_inputs(n: int, model_cens: str, cens_par: float, covar: float):
    """
    Validates input parameters for CPHM data generation.

    Parameters:
    - n (int): Number of data points to generate.
    - model_cens (str): Censoring model, must be "uniform" or "exponential".
    - cens_par (float): Parameter for the censoring model, must be > 0.
    - covar (float): Covariate value, must be > 0.

    Raises:
    - ValueError: If any input is invalid.
    """
    if n <= 0:
        raise ValueError("Argument 'n' must be greater than 0")
    if model_cens not in {"uniform", "exponential"}:
        raise ValueError("Argument 'model_cens' must be one of 'uniform' or 'exponential'")
    if cens_par <= 0:
        raise ValueError("Argument 'cens_par' must be greater than 0")
    if covar <= 0:
        raise ValueError("Argument 'covar' must be greater than 0")

def validate_gen_cmm_inputs(n: int, model_cens: str, cens_par: float, beta: list, covar: float, rate: list):
    """
    Validate inputs for generating CMM (Continuous-Time Markov Model) data.

    Parameters:
    - n (int): Number of individuals.
    - model_cens (str): Censoring model, must be "uniform" or "exponential".
    - cens_par (float): Parameter for censoring distribution, must be > 0.
    - beta (list): Regression coefficients, must have length 3.
    - covar (float): Covariate value, must be > 0.
    - rate (list): Transition rates, must have length 6.

    Raises:
    - ValueError: If any parameter is invalid.
    """
    if n <= 0:
        raise ValueError("Argument 'n' must be greater than 0")
    if model_cens not in {"uniform", "exponential"}:
        raise ValueError("Argument 'model_cens' must be one of 'uniform' or 'exponential'")
    if cens_par <= 0:
        raise ValueError("Argument 'cens_par' must be greater than 0")
    if len(beta) != 3:
        raise ValueError("Argument 'beta' must be a list of length 3")
    if covar <= 0:
        raise ValueError("Argument 'covar' must be greater than 0")
    if len(rate) != 6:
        raise ValueError("Argument 'rate' must be a list of length 6")
