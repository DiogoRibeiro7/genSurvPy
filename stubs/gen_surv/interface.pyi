from gen_surv.aft import gen_aft_log_normal as gen_aft_log_normal
from gen_surv.cmm import gen_cmm as gen_cmm
from gen_surv.cphm import gen_cphm as gen_cphm
from gen_surv.tdcm import gen_tdcm as gen_tdcm
from gen_surv.thmm import gen_thmm as gen_thmm

def generate(model: str, **kwargs): ...
