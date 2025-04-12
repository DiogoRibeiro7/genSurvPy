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
        raise ValueError(
            "Argument 'model_cens' must be one of 'uniform' or 'exponential'")
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
        raise ValueError(
            "Argument 'model_cens' must be one of 'uniform' or 'exponential'")
    if cens_par <= 0:
        raise ValueError("Argument 'cens_par' must be greater than 0")
    if len(beta) != 3:
        raise ValueError("Argument 'beta' must be a list of length 3")
    if covar <= 0:
        raise ValueError("Argument 'covar' must be greater than 0")
    if len(rate) != 6:
        raise ValueError("Argument 'rate' must be a list of length 6")


def validate_gen_tdcm_inputs(n: int, dist: str, corr: float, dist_par: list,
                             model_cens: str, cens_par: float, beta: list, lam: float):
    """
    Validate inputs for generating TDCM (Time-Dependent Covariate Model) data.

    Parameters:
    - n (int): Number of observations.
    - dist (str): "weibull" or "exponential".
    - corr (float): Correlation coefficient.
    - dist_par (list): Distribution parameters.
    - model_cens (str): "uniform" or "exponential".
    - cens_par (float): Censoring parameter.
    - beta (list): Length-2 list of regression coefficients.
    - lam (float): Lambda parameter, must be > 0.

    Raises:
    - ValueError: For any invalid input.
    """
    if n <= 0:
        raise ValueError("Argument 'n' must be greater than 0")

    if dist not in {"weibull", "exponential"}:
        raise ValueError(
            "Argument 'dist' must be one of 'weibull' or 'exponential'")

    if dist == "weibull":
        if not (0 < corr <= 1):
            raise ValueError("With dist='weibull', 'corr' must be in (0,1]")
        if len(dist_par) != 4 or any(p <= 0 for p in dist_par):
            raise ValueError(
                "With dist='weibull', 'dist_par' must be a positive list of length 4")

    if dist == "exponential":
        if not (-1 <= corr <= 1):
            raise ValueError(
                "With dist='exponential', 'corr' must be in [-1,1]")
        if len(dist_par) != 2 or any(p <= 0 for p in dist_par):
            raise ValueError(
                "With dist='exponential', 'dist_par' must be a positive list of length 2")

    if model_cens not in {"uniform", "exponential"}:
        raise ValueError(
            "Argument 'model_cens' must be one of 'uniform' or 'exponential'")

    if cens_par <= 0:
        raise ValueError("Argument 'cens_par' must be greater than 0")

    if not isinstance(beta, list) or len(beta) != 2:
        raise ValueError("Argument 'beta' must be a list of length 2")

    if lam <= 0:
        raise ValueError("Argument 'lambda' must be greater than 0")


def validate_gen_thmm_inputs(n: int, model_cens: str, cens_par: float, beta: list, covar: float, rate: list):
    """
    Validate inputs for generating THMM (Time-Homogeneous Markov Model) data.

    Parameters:
    - n (int): Number of samples, must be > 0.
    - model_cens (str): Must be "uniform" or "exponential".
    - cens_par (float): Must be > 0.
    - beta (list): List of length 3 (regression coefficients).
    - covar (float): Positive covariate value.
    - rate (list): List of length 3 (transition rates).

    Raises:
    - ValueError if any input is invalid.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Argument 'n' must be a positive integer.")

    if model_cens not in {"uniform", "exponential"}:
        raise ValueError(
            "Argument 'model_cens' must be one of 'uniform' or 'exponential'")

    if not isinstance(cens_par, (int, float)) or cens_par <= 0:
        raise ValueError("Argument 'cens_par' must be a positive number.")

    if not isinstance(beta, list) or len(beta) != 3:
        raise ValueError("Argument 'beta' must be a list of length 3.")

    if not isinstance(covar, (int, float)) or covar <= 0:
        raise ValueError("Argument 'covar' must be greater than 0.")

    if not isinstance(rate, list) or len(rate) != 3:
        raise ValueError("Argument 'rate' must be a list of length 3.")
