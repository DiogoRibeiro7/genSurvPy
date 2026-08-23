# Plotting

Four helpers built on matplotlib and lifelines. All of them return
`(figure, axes)`, so you can keep styling, annotating and saving afterwards.

Both libraries are hard dependencies — nothing extra to install.

## Kaplan-Meier curves

```python
import matplotlib.pyplot as plt
from gen_surv import generate, plot_survival_curve

df = generate(model="cphm", n=2000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)

fig, ax = plot_survival_curve(df, title="CPHM, n=2000")
fig.savefig("km.png", dpi=200, bbox_inches="tight")
```

| Parameter | Default | Effect |
|---|---|---|
| `time_col`, `status_col` | `"time"`, `"status"` | column names |
| `group_col` | `None` | draw one curve per level of this column |
| `confidence_intervals` | `True` | shade pointwise confidence bands |
| `title` | `"Kaplan-Meier Survival Curve"` | plot title |
| `figsize` | `(10, 6)` | figure size in inches |
| `ci_alpha` | `0.2` | opacity of the confidence band |

### Stratified curves

`group_col` needs a discrete column. Generated covariates are usually
continuous, so bin one first:

```python
import pandas as pd

df["X0_group"] = pd.qcut(df["X0"], q=3, labels=["low", "mid", "high"])
fig, ax = plot_survival_curve(df, group_col="X0_group",
                              title="Survival by covariate tertile")
```

Or use a model with a binary covariate — see
[Covariates](covariates.md#configurable-models).

## Covariate effects

[`plot_covariate_effect`](../api/analysis.md#gen_surv.visualization.plot_covariate_effect)
does that binning for you:

```python
from gen_surv import plot_covariate_effect

fig, ax = plot_covariate_effect(df, covariate_col="X0", n_groups=3)
```

| Parameter | Default | Effect |
|---|---|---|
| `covariate_col` | — | the continuous covariate to split |
| `n_groups` | `3` | number of equal-sized groups |
| `title` | `"Effect of Covariate on Survival"` | plot title |

With a positive `beta`, higher covariate values mean a higher hazard, so the
curves should fan out in order. If they cross or fail to separate, either the
effect is small relative to `n` or something upstream is wrong — which is
exactly the kind of check simulated data is for.

## Comparing models

[`plot_hazard_comparison`](../api/analysis.md#gen_surv.visualization.plot_hazard_comparison)
takes a dict of datasets and overlays their smoothed hazards:

```python
from gen_surv import generate, plot_hazard_comparison

models = {
    f"shape={s}": generate(model="aft_weibull", n=2000, beta=[0.5, -0.3],
                           shape=s, scale=2.0, model_cens="uniform",
                           cens_par=5.0, seed=42)
    for s in (0.5, 1.0, 2.0)
}

fig, ax = plot_hazard_comparison(models, title="Weibull hazard by shape",
                                 bandwidth=0.5)
```

The falling, flat and rising hazards should be visibly distinct. `bandwidth`
controls the kernel smoother: too small is noisy, too large flattens real
structure.

This is also the way to see the difference between families — put a `cphm`
sample, a `aft_log_logistic` sample and a `piecewise_exponential` sample in one
dict.

## Descriptive statistics

[`describe_survival`](../api/analysis.md#gen_surv.visualization.describe_survival)
returns a frame rather than a plot, and pairs naturally with these figures. See
[Summarising a dataset](summaries.md).

## Practical notes

**Saving instead of showing.** In scripts and CI there is no display; use a
non-interactive backend and save:

```python
import matplotlib
matplotlib.use("Agg")
```

**Styling after the fact.** The returned axes is a normal matplotlib object:

```python
fig, ax = plot_survival_curve(df)
ax.set_xlabel("Years since randomisation")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
```

**Several panels.** The helpers create their own figure, so to arrange them
side by side, save separately or re-plot onto your own axes with lifelines
directly.

**Column names.** For `tdcm`, pass `time_col="stop"`. For `thmm` there is no
single event indicator — derive one first, as in [THMM](../models/thmm.md).

## From the command line

```bash
gen_surv visualize data.csv --time-col time --status-col status \
    --group-col X0 --output km.png
```

See [Command line](cli.md).

## Related

- [Summarising a dataset](summaries.md)
- API: [Analysis](../api/analysis.md)
