import argparse
import pandas as pd
from gen_surv.cphm import gen_cphm
from gen_surv.cmm import gen_cmm
from gen_surv.tdcm import gen_tdcm
from gen_surv.thmm import gen_thmm

def run_example(model: str):
    if model == "cphm":
        df = gen_cphm(n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)
    elif model == "cmm":
        df = gen_cmm(n=10, model_cens="exponential", cens_par=1.0,
                     beta=[0.5, 0.2, -0.1], covar=2.0, rate=[0.1, 1.0, 0.2, 1.0, 0.3, 1.0])
    elif model == "tdcm":
        df = gen_tdcm(n=10, dist="weibull", corr=0.5, dist_par=[1, 2, 1, 2],
                      model_cens="uniform", cens_par=0.5, beta=[0.1, 0.2, 0.3], lam=1.0)
    elif model == "thmm":
        df = gen_thmm(n=10, model_cens="uniform", cens_par=0.5,
                      beta=[0.1, 0.2, 0.3], covar=1.0, rate=[0.5, 0.6, 0.7])
    else:
        raise ValueError(f"Unknown model: {model}")
    
    print(df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gen_surv model example.")
    parser.add_argument("model", choices=["cphm", "cmm", "tdcm", "thmm"],
                        help="Model to run (cphm, cmm, tdcm, thmm)")
    args = parser.parse_args()
    run_example(args.model)
