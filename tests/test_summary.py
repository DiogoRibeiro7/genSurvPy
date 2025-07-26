from gen_surv import generate
from gen_surv.summary import summarize_survival_dataset


def test_summarize_survival_dataset_basic():
    df = generate(model="cphm", n=20, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
    summary = summarize_survival_dataset(df, verbose=False)
    assert isinstance(summary, dict)
    assert "dataset_info" in summary
