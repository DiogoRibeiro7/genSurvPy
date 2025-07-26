def validate_gen_cphm_inputs(n: int, model_cens: str, cens_par: float, covariate_range: float):
    """
    Validates input parameters for CPHM data generation.

    Parameters:
    - n (int): Number of data points to generate.
    - model_cens (str): Censoring model, must be "uniform" or "exponential".
    - cens_par (float): Parameter for the censoring model, must be > 0.
    - covariate_range (float): Upper bound for covariate values, must be > 0.

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
    if covariate_range <= 0:
        raise ValueError("Argument 'covariate_range' must be greater than 0")


def validate_gen_cmm_inputs(n: int, model_cens: str, cens_par: float, beta: list, covariate_range: float, rate: list):
    """
    Validate inputs for generating CMM (Continuous-Time Markov Model) data.

    Parameters:
    - n (int): Number of individuals.
    - model_cens (str): Censoring model, must be "uniform" or "exponential".
    - cens_par (float): Parameter for censoring distribution, must be > 0.
    - beta (list): Regression coefficients, must have length 3.
    - covariate_range (float): Upper bound for covariate values, must be > 0.
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
    if covariate_range <= 0:
        raise ValueError("Argument 'covariate_range' must be greater than 0")
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

    if not isinstance(beta, list) or len(beta) != 3:
        raise ValueError("Argument 'beta' must be a list of length 3")

    if lam <= 0:
        raise ValueError("Argument 'lambda' must be greater than 0")


def validate_gen_thmm_inputs(n: int, model_cens: str, cens_par: float, beta: list, covariate_range: float, rate: list):
    """
    Validate inputs for generating THMM (Time-Homogeneous Markov Model) data.

    Parameters:
    - n (int): Number of samples, must be > 0.
    - model_cens (str): Must be "uniform" or "exponential".
    - cens_par (float): Must be > 0.
    - beta (list): List of length 3 (regression coefficients).
    - covariate_range (float): Positive upper bound for covariate values.
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

    if not isinstance(covariate_range, (int, float)) or covariate_range <= 0:
        raise ValueError("Argument 'covariate_range' must be greater than 0.")

    if not isinstance(rate, list) or len(rate) != 3:
        raise ValueError("Argument 'rate' must be a list of length 3.")


def validate_dg_biv_inputs(n: int, dist: str, corr: float, dist_par: list):
    """
    Validate inputs for the sample_bivariate_distribution function.

    Parameters:
    - n (int): Number of samples to generate.
    - dist (str): Must be "weibull" or "exponential".
    - corr (float): Must be between -1 and 1.
    - dist_par (list): Must contain positive values, and correct length for the distribution.

    Raises:
    - ValueError if any input is invalid.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Argument 'n' must be a positive integer.")

    if dist not in {"weibull", "exponential"}:
        raise ValueError("Argument 'dist' must be one of 'weibull' or 'exponential'.")

    if not isinstance(corr, (int, float)) or not (-1 < corr < 1):
        raise ValueError("Argument 'corr' must be a numeric value between -1 and 1.")

    if not isinstance(dist_par, list) or len(dist_par) == 0:
        raise ValueError("Argument 'dist_par' must be a non-empty list of positive values.")

    if any(p <= 0 for p in dist_par):
        raise ValueError("All elements in 'dist_par' must be greater than 0.")

    if dist == "exponential" and len(dist_par) != 2:
        raise ValueError("Exponential distribution requires exactly 2 positive parameters.")

    if dist == "weibull" and len(dist_par) != 4:
        raise ValueError("Weibull distribution requires exactly 4 positive parameters.")


def validate_gen_aft_log_normal_inputs(n, beta, sigma, model_cens, cens_par):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(beta, (list, tuple)) or not all(isinstance(b, (int, float)) for b in beta):
        raise ValueError("beta must be a list of numbers")

    if not isinstance(sigma, (int, float)) or sigma <= 0:
        raise ValueError("sigma must be a positive number")

    if model_cens not in ("uniform", "exponential"):
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    if not isinstance(cens_par, (int, float)) or cens_par <= 0:
        raise ValueError("cens_par must be a positive number")


def validate_gen_aft_weibull_inputs(n, beta, shape, scale, model_cens, cens_par):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(beta, (list, tuple)) or not all(isinstance(b, (int, float)) for b in beta):
        raise ValueError("beta must be a list of numbers")

    if not isinstance(shape, (int, float)) or shape <= 0:
        raise ValueError("shape must be a positive number")

    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("scale must be a positive number")

    if model_cens not in ("uniform", "exponential"):
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    if not isinstance(cens_par, (int, float)) or cens_par <= 0:
        raise ValueError("cens_par must be a positive number")


def validate_gen_aft_log_logistic_inputs(n, beta, shape, scale, model_cens, cens_par):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(beta, (list, tuple)) or not all(isinstance(b, (int, float)) for b in beta):
        raise ValueError("beta must be a list of numbers")

    if not isinstance(shape, (int, float)) or shape <= 0:
        raise ValueError("shape must be a positive number")

    if not isinstance(scale, (int, float)) or scale <= 0:
        raise ValueError("scale must be a positive number")

    if model_cens not in ("uniform", "exponential"):
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    if not isinstance(cens_par, (int, float)) or cens_par <= 0:
        raise ValueError("cens_par must be a positive number")


def validate_competing_risks_inputs(n, n_risks, baseline_hazards, betas, model_cens, cens_par):
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(n_risks, int) or n_risks <= 0:
        raise ValueError("n_risks must be a positive integer")

    if baseline_hazards is not None and (
        not isinstance(baseline_hazards, (list, tuple)) or
        len(baseline_hazards) != n_risks or
        any(h <= 0 for h in baseline_hazards)
    ):
        raise ValueError("baseline_hazards must be a list of positive numbers with length n_risks")

    if betas is not None and (
        not isinstance(betas, list) or
        any(not isinstance(b, list) for b in betas)
    ):
        raise ValueError("betas must be a list of lists")

    if model_cens not in ("uniform", "exponential"):
        raise ValueError("model_cens must be 'uniform' or 'exponential'")

    if not isinstance(cens_par, (int, float)) or cens_par <= 0:
        raise ValueError("cens_par must be a positive number")
