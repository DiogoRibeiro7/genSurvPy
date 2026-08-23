# Time-dependent covariates

`model="tdcm"` — a Cox model where one covariate **switches value during
follow-up**. Think of a treatment that starts partway through, or an exposure
that begins at some point after enrolment.

## The model

Each subject has a baseline covariate $Z_1$ and a crossover time drawn jointly
from a bivariate distribution, so the two can be correlated. Before the
crossover the hazard is

$$
h(t) = \lambda \exp(\beta_0 Z_1),
$$

and after it the time-dependent covariate switches on, multiplying the hazard by
$\exp(\beta_1)$:

$$
h(t) = \lambda \exp(\beta_0 Z_1 + \beta_1).
$$

The event time is drawn by inversion, splitting on whether the event falls
before or after the crossover. The recorded `tdcov` is the covariate's value
**at the moment the subject left the study** — 0 if the event or censoring came
before the switch, 1 if after.

## Parameters

```python
gen_tdcm(n, dist, corr, dist_par, model_cens, cens_par, beta, lam, seed=None)
```

| Parameter | Type | Constraint | Meaning |
|---|---|---|---|
| `n` | `int` | > 0 | number of subjects |
| `dist` | `str` | `"weibull"` or `"exponential"` | marginals of the bivariate covariate draw |
| `corr` | `float` | `(0, 1]` for Weibull, `[-1, 1]` for exponential | dependence between baseline covariate and crossover time |
| `dist_par` | `Sequence[float]` | **4** for Weibull, **2** for exponential; all > 0 | parameters of those marginals |
| `model_cens` | `str` | `"uniform"` or `"exponential"` | censoring mechanism |
| `cens_par` | `float` | > 0 | censoring parameter |
| `beta` | `Sequence[float]` | **exactly 2** | `beta[0]` baseline covariate effect, `beta[1]` effect of the covariate switching on |
| `lam` | `float` | > 0 | baseline hazard rate |
| `seed` | `int` \| `Generator` \| `None` | — | reproducibility |

!!! warning "`beta` takes two coefficients, not three"

    Releases up to 1.2.0 required three and silently ignored the third. Passing
    three still works but raises a `DeprecationWarning`, and will become an
    error:

    ```text
    DeprecationWarning: gen_tdcm uses two coefficients; passing three is
    deprecated because the third is ignored, and it will raise in a future
    release.
    ```

## Example

```python
from gen_surv import generate

df = generate(model="tdcm", n=6, dist="weibull", corr=0.5,
              dist_par=[1.0, 2.0, 1.0, 2.0], model_cens="uniform",
              cens_par=5.0, beta=[0.5, 0.3], lam=1.0, seed=42)
print(df)
```

```text
 id  start     stop  status  covariate  tdcov
1.0    0.0 0.806478     1.0   0.494017    0.0
2.0    0.0 0.784498     1.0   0.748251    1.0
3.0    0.0 0.294102     1.0   1.378558    0.0
4.0    0.0 0.180955     1.0   0.707759    0.0
5.0    0.0 0.585859     1.0   0.644816    0.0
6.0    0.0 0.047366     1.0   0.661836    0.0
```

| Column | Meaning |
|---|---|
| `id` | subject, from 1 — stored as `float64` |
| `start`, `stop` | the observation interval; `start` is always 0 |
| `status` | `1.0` event at `stop`, `0.0` censored |
| `covariate` | the baseline covariate $Z_1$ — note it is not called `X0` |
| `tdcov` | `1.0` if the time-dependent covariate had switched on by `stop`, else `0.0` |

## What the output does and does not give you

`start` is always 0 and each subject has one row, so **the frame does not split
the risk interval at the crossover** — and the crossover time itself is not
reported. `tdcov` records only the value in force when the subject left.

That has a consequence worth being explicit about: **fitting a Cox model to
this frame with `tdcov` as an ordinary covariate is biased**, in the classic
time-dependent-covariate way. Subjects can only be observed with `tdcov = 1` if
they survived long enough to switch, so the switched group looks artificially
healthy. With `beta = [0.5, 0.3]` and `n = 40000`:

```python
from lifelines import CoxPHFitter

d = df.rename(columns={"stop": "time"})
CoxPHFitter().fit(d[["time", "status", "covariate", "tdcov"]],
                  duration_col="time", event_col="status").params_
```

```text
covariate    0.409
tdcov       -0.421
```

The baseline effect is pulled down from 0.5, and `tdcov` comes out at −0.42
against a true **+0.3** — the sign is reversed. This is the bias the model is
there to demonstrate, not an estimate to trust.

To fit it properly you would need the crossover time so you could split each
subject into `(0, switch]` with `tdcov = 0` and `(switch, stop]` with
`tdcov = 1`. The generator does not currently expose that time, so use this
model to **produce** time-dependent-covariate data and to demonstrate what goes
wrong when it is analysed naively, rather than as a benchmark for recovering
`beta[1]`.

## Correlation between covariate and crossover

`corr` controls how strongly the baseline covariate and the crossover time move
together — whether the subjects who switch early are also the high-risk ones:

```python
for corr in (0.1, 0.5, 0.9):
    df = generate(model="tdcm", n=5000, dist="weibull", corr=corr,
                  dist_par=[1.0, 2.0, 1.0, 2.0], model_cens="uniform",
                  cens_par=5.0, beta=[0.5, 0.3], lam=1.0, seed=1)
    print(f"corr={corr}  switched before leaving: {df['tdcov'].mean():.3f}")
```

The underlying draw comes from
[`sample_bivariate_distribution`](../api/censoring.md#gen_surv.bivariate.sample_bivariate_distribution),
which you can call directly if you want the covariates without the survival
times.

## Related

- [Cox proportional hazards](cphm.md) — the fixed-covariate version
- [Illness-death, intervals (CMM)](cmm.md) — when the *state* changes rather than a covariate
- [Output schemas](../getting-started/schemas.md#tdcm)
- API: [`gen_tdcm`](../api/generators.md#gen_surv.tdcm.gen_tdcm)
