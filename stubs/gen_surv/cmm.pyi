from gen_surv.censoring import rexpocens as rexpocens, runifcens as runifcens
from gen_surv.validate import validate_gen_cmm_inputs as validate_gen_cmm_inputs

def generate_event_times(z1: float, beta: list, rate: list) -> dict: ...
def gen_cmm(n, model_cens, cens_par, beta, covar, rate): ...
