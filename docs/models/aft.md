# Accelerated failure time

Three generators — `aft_ln`, `aft_weibull` and `aft_log_logistic` — where
covariates act on the event **time** rather than on the hazard.

Use them when the proportional-hazards assumption is what you want to violate,
or when a time-scale interpretation ("this treatment doubles survival") is more
natural than a hazard ratio.

!!! danger "`beta` does not mean the same thing in all three"

    Only `aft_ln` puts `beta` directly on log time. In `aft_weibull` and
    `aft_log_logistic`, `beta` enters through `exp(-eta / shape)`, so its
    effect on log time is $-\beta/\texttt{shape}$ — different sign, different
    magnitude. [The section below](#what-beta-actually-does) shows the
    measurements. Getting this wrong will make a correct estimator look broken.

## The three models

### Log-normal — `aft_ln`

$$
\log T_i = X_i^\top \beta + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

so $T_i$ is log-normal with median $\exp(X_i^\top\beta)$. The survival function
is

$$
S(t \mid X) = 1 - \Phi\!\left(\frac{\log t - X^\top \beta}{\sigma}\right).
$$

The hazard rises then falls — useful precisely because it is not proportional.

### Weibull — `aft_weibull`

$$
T_i = \texttt{scale} \cdot \big(-\log U_i \cdot e^{-\eta_i}\big)^{1/\texttt{shape}},
\qquad U_i \sim \mathrm{Uniform}(0,1), \quad \eta_i = X_i^\top\beta
$$

The Weibull is the one distribution that is both proportional-hazards and
accelerated-failure-time, and this parameterisation is the **PH** one: $\beta$
is a log hazard ratio. The hazard is monotone — falling for
`shape < 1`, constant at `shape = 1`, rising for `shape > 1`.

### Log-logistic — `aft_log_logistic`

$$
T_i = \texttt{scale} \cdot \left(\frac{U_i}{1 - U_i}\right)^{1/\texttt{shape}} e^{-\eta_i/\texttt{shape}}
$$

A unimodal hazard: it rises to a peak, then decays. The standard choice when
risk is highest some way into follow-up.

## Parameters

```python
gen_aft_log_normal(n, beta, sigma, model_cens, cens_par, seed=None)
gen_aft_weibull(n, beta, shape, scale, model_cens, cens_par, seed=None)
gen_aft_log_logistic(n, beta, shape, scale, model_cens, cens_par, seed=None)
```

| Parameter | Type | Applies to | Meaning |
|---|---|---|---|
| `n` | `int` | all | number of subjects |
| `beta` | `list[float]` | all | one coefficient per covariate; **its length sets the number of covariates** |
| `sigma` | `float` | `aft_ln` | standard deviation of the log-time error |
| `shape` | `float` | `aft_weibull`, `aft_log_logistic` | shape parameter — controls the hazard's direction |
| `scale` | `float` | `aft_weibull`, `aft_log_logistic` | time scale |
| `model_cens` | `str` | all | `"uniform"` or `"exponential"` |
| `cens_par` | `float` | all | censoring parameter |
| `seed` | `int` \| `Generator` \| `None` | all | reproducibility |

Covariates are drawn as independent standard normals, `X0 … X{p-1}` where
`p = len(beta)`. Pass `beta=[0.5, -0.3]` and you get two covariates; pass five
coefficients and you get five.

## Example

```python
from gen_surv import generate

df = generate(model="aft_ln", n=6, beta=[0.5, -0.3], sigma=1.0,
              model_cens="exponential", cens_par=3.0, seed=42)
print(df)
```

```text
 id     time  status        X0        X1
  0 1.699586       1  0.304717 -1.039984
  1 1.238956       0  0.750451  0.940565
  2 0.889270       1 -1.951035 -1.302180
  3 0.496337       1  0.127840 -0.316243
  4 1.851995       1 -0.016801 -0.853044
  5 0.471177       1  0.879398  0.777792
```

## What `beta` actually does

Generate with almost no censoring and regress `log(time)` on the covariates.
For `beta = [0.5, -0.3]`, `shape = 1.5`, `n = 40000`:

| Model | Fitted effect on `log(time)` | Matches |
|---|---|---|
| `aft_ln` | `X0` 0.500, `X1` −0.298 | $\beta$ itself — `[0.5, -0.3]` |
| `aft_weibull` | `X0` −0.332, `X1` 0.209 | $-\beta/\texttt{shape}$ — `[-0.333, 0.200]` |
| `aft_log_logistic` | `X0` −0.336, `X1` 0.187 | $-\beta/\texttt{shape}$ — `[-0.333, 0.200]` |

For `aft_weibull` the natural estimator is therefore a Cox model, which reads
`beta` off directly:

```python
from gen_surv import generate
from lifelines import CoxPHFitter

df = generate(model="aft_weibull", n=40000, beta=[0.5, -0.3], shape=1.5,
              scale=2.0, model_cens="uniform", cens_par=1e6, seed=7)

cph = CoxPHFitter().fit(df[["time", "status", "X0", "X1"]],
                        duration_col="time", event_col="status")
print(cph.params_.round(3).to_dict())
```

```text
{'X0': 0.499, 'X1': -0.311}
```

And for `aft_ln`, ordinary least squares on log time recovers `beta` and
`sigma`:

```python
import numpy as np
from gen_surv import generate

df = generate(model="aft_ln", n=20000, beta=[0.5, -0.3], sigma=1.0,
              model_cens="exponential", cens_par=100.0, seed=7)

X = np.column_stack([df["X0"], df["X1"]])
coef, *_ = np.linalg.lstsq(X, np.log(df["time"]), rcond=None)
resid_sd = np.std(np.log(df["time"]) - X @ coef)
print(coef.round(3), round(float(resid_sd), 3))
```

```text
[ 0.491 -0.299] 1.002
```

(That shortcut ignores the 1.8% of rows that are censored. With heavier
censoring, fit a proper AFT model — `lifelines.LogNormalAFTFitter` — instead of
regressing on observed times.)

## Choosing `shape`

```python
from gen_surv import generate, plot_hazard_comparison

models = {
    f"shape={s}": generate(model="aft_weibull", n=2000, beta=[0.5, -0.3],
                           shape=s, scale=2.0, model_cens="uniform",
                           cens_par=5.0, seed=42)
    for s in (0.5, 1.0, 2.0)
}
fig, ax = plot_hazard_comparison(models, title="Weibull hazard by shape")
```

| `shape` | Hazard over time | Typical use |
|---|---|---|
| < 1 | falling | early failures, infant mortality |
| = 1 | constant | equivalent to an exponential model |
| > 1 | rising | wear-out, ageing |

## Related

- [Output schemas](../getting-started/schemas.md#aft-models) — the exact frame
- [Cox proportional hazards](cphm.md) — when you want PH to hold
- [Plotting](../guides/plotting.md) — `plot_hazard_comparison` used above
- API: [`gen_aft_log_normal`, `gen_aft_weibull`, `gen_aft_log_logistic`](../api/generators.md#gen_surv.aft)
