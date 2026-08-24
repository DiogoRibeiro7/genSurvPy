# Mixture cure

`model="mixture_cure"` — a fraction of the population is **cured** and will
never experience the event, no matter how long you watch. The rest fail
normally.

Use it for long-term-survivor data: a Kaplan-Meier curve that flattens onto a
plateau well above zero and stays there.

## The model

Two components. First, whether a subject is cured, through a logistic model:

$$
\Pr(\text{cured} \mid X) =
\frac{\exp(\alpha_0 + X^\top \beta_{\text{cure}})}{1 + \exp(\alpha_0 + X^\top \beta_{\text{cure}})},
$$

where $\alpha_0$ is set from the `cure_fraction` you ask for. Then, for the
uncured, an exponential failure time whose hazard is scaled by the covariates:

$$
T_i \mid \text{uncured} \sim \mathrm{Exponential}\big(\lambda \exp(X_i^\top\beta_{\text{surv}})\big).
$$

Cured subjects are censored at `max_time` or at their random censoring time,
whichever comes first — from the outside they are indistinguishable from
someone who simply had not failed yet. That is exactly the inference problem a
cure model has to solve, and the generated frame hands you the answer key in
the `cured` column.

## Parameters

```python
gen_mixture_cure(n, cure_fraction, baseline_hazard=0.5, betas_survival=None,
                 betas_cure=None, n_covariates=2, covariate_dist="normal",
                 covariate_params=None, model_cens="uniform", cens_par=5.0,
                 max_time=10.0, seed=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | — | number of subjects |
| `cure_fraction` | `float` | — | target proportion cured, in (0, 1) |
| `baseline_hazard` | `float` | `0.5` | exponential hazard for the uncured |
| `betas_survival` | `list[float]` \| `None` | `None` | covariate effects on the uncured failure hazard |
| `betas_cure` | `list[float]` \| `None` | `None` | covariate effects on the probability of being cured |
| `n_covariates` | `int` | `2` | number of covariates when the `betas_*` are omitted |
| `covariate_dist` | `"normal"` \| `"uniform"` \| `"binary"` | `"normal"` | see [Covariates](../guides/covariates.md) |
| `covariate_params` | `dict` \| `None` | `None` | distribution parameters |
| `model_cens` | `str` | `"uniform"` | censoring mechanism |
| `cens_par` | `float` | `5.0` | censoring parameter |
| `max_time` | `float` \| `None` | `10.0` | administrative censoring |
| `seed` | `int` \| `Generator` \| `None` | `None` | reproducibility |

With `betas_cure=[0.0, 0.0]` the cure probability is the same for everyone and
equals `cure_fraction` exactly. With non-zero values it varies by subject, and
`cure_fraction` becomes the population average rather than an individual
probability.

## Example

```python
from gen_surv import generate

df = generate(model="mixture_cure", n=6, cure_fraction=0.3, seed=42)
print(df)
```

```text
   id      time  status  cured        X0        X1
0   0  0.973194       0      0 -1.951035 -1.302180
1   1  2.041874       1      0  0.127840 -0.316243
2   2  0.219019       0      1 -0.016801 -0.853044
3   3  0.771447       0      1  0.879398  0.777792
4   4  3.415245       0      1  0.066031  1.127241
5   5  0.781287       1      0  0.467509 -0.859292
```

The `cured` column is **ground truth you would never have in real data**. Every
cured subject has `status == 0`; the converse is not true, because an uncured
subject can also be censored before failing.

```python
import pandas as pd

pd.crosstab(df["cured"], df["status"])
```

## Check: is the cure fraction the one you asked for?

```python
from gen_surv import generate, cure_fraction_estimate

for target in (0.2, 0.4, 0.6):
    df = generate(model="mixture_cure", n=20000, cure_fraction=target,
                  baseline_hazard=1.0, betas_cure=[0.0, 0.0],
                  betas_survival=[0.0, 0.0], model_cens="uniform",
                  cens_par=20.0, max_time=20.0, seed=7)
    print(f"target={target}  actually cured={df['cured'].mean():.3f}  "
          f"estimated={cure_fraction_estimate(df):.3f}")
```

```text
target=0.2  actually cured=0.197  estimated=0.197
target=0.4  actually cured=0.398  estimated=0.399
target=0.6  actually cured=0.602  estimated=0.602
```

[`cure_fraction_estimate`](../api/generators.md#gen_surv.mixture.cure_fraction_estimate)
reads the plateau off the tail of the Kaplan-Meier curve, so it needs follow-up
long enough for the curve to flatten — here `max_time=20` against a hazard of
1.0. Shorten the follow-up and the estimate drifts upward, because uncured
survivors are still mixed in with the cured.

## Seeing the plateau

```python
from gen_surv import generate, plot_survival_curve

df = generate(model="mixture_cure", n=2000, cure_fraction=0.4,
              baseline_hazard=1.0, betas_cure=[0.0, 0.0],
              betas_survival=[0.0, 0.0], cens_par=20.0, max_time=20.0, seed=1)

fig, ax = plot_survival_curve(df, title="Cure fraction 0.4")
```

The curve drops steeply while the uncured fail, then flattens at roughly 0.4 and
stays there. A survival model without a cure component will insist the curve
must eventually reach zero and will fit the tail badly — which is the point of
generating this data.

## Related

- [Competing risks](competing-risks.md) — several ways to fail, rather than not failing
- [Plotting](../guides/plotting.md) — Kaplan-Meier curves
- API: [`gen_mixture_cure`, `cure_fraction_estimate`](../api/generators.md#gen_surv.mixture)
