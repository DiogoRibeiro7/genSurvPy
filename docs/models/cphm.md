# Cox proportional hazards

`model="cphm"` — the textbook proportional-hazards setup: one covariate, one
coefficient, an exponential baseline hazard.

Use it when you want a dataset whose **log hazard ratio you know**, to check
that a Cox implementation recovers it.

## The model

The hazard for subject $i$ with covariate $X_i$ is

$$
h(t \mid X_i) = h_0(t) \exp(\beta X_i),
$$

with a constant baseline $h_0(t) = 1$. Event times therefore come from an
exponential distribution whose rate is scaled by the covariate:

$$
T_i \mid X_i \sim \mathrm{Exponential}\big(\exp(\beta X_i)\big),
\qquad X_i \sim \mathrm{Uniform}(0, \texttt{covariate\_range}).
$$

A censoring time $C_i$ is drawn independently, and the row records
$\min(T_i, C_i)$ together with which came first.

Because the baseline is constant, the hazard does not change over follow-up.
For a hazard that rises or falls, use [Weibull AFT](aft.md); for an arbitrary
shape, [piecewise exponential](piecewise-exponential.md).

## Parameters

```python
gen_cphm(n, model_cens, cens_par, beta, covariate_range, seed=None)
```

| Parameter | Type | Constraint | Meaning |
|---|---|---|---|
| `n` | `int` | > 0 | number of subjects, one row each |
| `model_cens` | `str` | `"uniform"` or `"exponential"` | censoring mechanism |
| `cens_par` | `float` | > 0 | censoring parameter — see [Censoring](../guides/censoring.md) |
| `beta` | `float` | — | log hazard ratio per unit of `X0`. A **scalar**, not a list |
| `covariate_range` | `float` | > 0 | `X0` is drawn from `Uniform(0, covariate_range)` |
| `seed` | `int` \| `Generator` \| `None` | — | reproducibility |

!!! note "`beta` is scalar here"

    `cphm` is the only model whose `beta` is a single number. AFT models take a
    list of coefficients, and the multi-state models take exactly three.

## Example

```python
from gen_surv import generate

df = generate(model="cphm", n=6, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)
print(df)
```

```text
       time  status        X0
0  0.438878     0.0  1.547912
1  0.094177     0.0  1.394736
2  0.037041     1.0  1.522279
3  0.370798     0.0  0.900772
4  0.646901     1.0  1.287730
5  0.251113     1.0  0.454477
```

| Column | dtype | Meaning |
|---|---|---|
| `time` | `float64` | `min(event, censoring)` |
| `status` | `float64` | `1.0` event, `0.0` censored |
| `X0` | `float64` | the covariate |

There is **no `id` column** — this is the only generator without one. See
[Output schemas](../getting-started/schemas.md).

## Controlling the censoring rate

With `model_cens="uniform"`, censoring times are `Uniform(0, cens_par)`, so a
larger `cens_par` censors less:

```python
for cens_par in (0.5, 1.0, 5.0, 100.0):
    df = generate(model="cphm", n=20000, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=cens_par, seed=1)
    print(f"cens_par={cens_par:>6}  event rate = {df['status'].mean():.3f}")
```

## Check: does a Cox model recover `beta`?

```python
from gen_surv import generate
from lifelines import CoxPHFitter

df = generate(model="cphm", n=5000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=7)

cph = CoxPHFitter().fit(df, duration_col="time", event_col="status")
print(cph.summary[["coef", "se(coef)", "coef lower 95%", "coef upper 95%"]].round(3))
```

```text
            coef  se(coef)  coef lower 95%  coef upper 95%
covariate
X0         0.501     0.035           0.433           0.569
```

At `n=20000` and five seeds the estimates are 0.502, 0.489, 0.500, 0.489, 0.517
— centred on the truth, and unaffected by how heavily the data is censored.

## Related

- [Censoring](../guides/censoring.md) — the mechanisms and how to hit a target event rate
- [AFT models](aft.md) — when the covariate should act on time rather than hazard
- [Time-dependent covariates](tdcm.md) — when the covariate changes during follow-up
- API: [`gen_cphm`](../api/generators.md#gen_surv.cphm.gen_cphm)
