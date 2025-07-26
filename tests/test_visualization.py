from gen_surv import generate
from gen_surv.visualization import plot_survival_curve


def test_plot_survival_curve_runs():
    df = generate(model="cphm", n=10, model_cens="uniform", cens_par=1.0, beta=0.5, covariate_range=2.0)
    fig, ax = plot_survival_curve(df)
    assert fig is not None
    assert ax is not None
