import pandas as pd
from gen_surv.censoring import rexpocens as rexpocens, runifcens as runifcens
from gen_surv.validate import validate_gen_cphm_inputs as validate_gen_cphm_inputs

def generate_cphm_data(n, rfunc, cens_par, beta, covariate_range): ...
def gen_cphm(n: int, model_cens: str, cens_par: float, beta: float, covar: float) -> pd.DataFrame: ...
