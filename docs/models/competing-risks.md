# Competing risks

`model="competing_risks"` and `model="competing_risks_weibull"` — several
distinct failure types, where the first one to occur is the one you observe and
the others are never seen for that subject.

Use them when "death" is not one thing: relapse versus death in remission,
device failure by mode, discharge versus in-hospital death.

## The model

Each cause $k$ has its own **cause-specific hazard**

$$
h_k(t \mid X) = h_{0k}(t) \exp(X^\top \beta_k),
$$

with an independent set of coefficients per cause. A latent time is drawn for
each cause and the smallest wins:

$$
T = \min_k T_k, \qquad \delta = \arg\min_k T_k .
$$

The observed `status` is the winning cause — `1`, `2`, … — or `0` if censoring
came first, or if nothing happened before `max_time`.

The two variants differ only in the baseline:

| Model | Baseline hazard | Parameters |
|---|---|---|
| `competing_risks` | constant, $h_{0k}(t) = \lambda_k$ | `baseline_hazards` |
| `competing_risks_weibull` | Weibull, monotone per cause | `shape_params`, `scale_params` |

## Parameters

```python
gen_competing_risks(n, n_risks=2, baseline_hazards=None, betas=None,
                    covariate_dist="normal", covariate_params=None,
                    max_time=10.0, model_cens="uniform", cens_par=5.0, seed=None)

gen_competing_risks_weibull(n, n_risks=2, shape_params=None, scale_params=None,
                            betas=None, covariate_dist="normal",
                            covariate_params=None, max_time=10.0,
                            model_cens="uniform", cens_par=5.0, seed=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | — | number of subjects |
| `n_risks` | `int` | `2` | number of competing causes |
| `baseline_hazards` | `list[float]` \| `None` | `None` | one constant hazard per cause (`competing_risks`) |
| `shape_params`, `scale_params` | `list[float]` \| `None` | `None` | one per cause (`competing_risks_weibull`) |
| `betas` | `list[list[float]]` \| `None` | `None` | **one row per cause**, one column per covariate |
| `covariate_dist` | `"normal"` \| `"uniform"` \| `"binary"` | `"normal"` | see [Covariates](../guides/covariates.md) |
| `covariate_params` | `dict` \| `None` | `None` | distribution parameters |
| `max_time` | `float` \| `None` | `10.0` | administrative censoring: nothing is observed beyond this |
| `model_cens` | `str` | `"uniform"` | random censoring mechanism |
| `cens_par` | `float` | `5.0` | censoring parameter |
| `seed` | `int` \| `Generator` \| `None` | `None` | reproducibility |

`betas` is a **matrix**: `betas[k][j]` is the effect of covariate `j` on cause
`k + 1`. Leave it out and coefficients are drawn at random, which is fine for a
smoke test and useless for validation.

## Example

Cause 1 driven by `X0`, cause 2 by `X1`, with cause 1 twice as fast:

```python
from gen_surv import generate

df = generate(model="competing_risks", n=20000, n_risks=2,
              baseline_hazards=[0.4, 0.2],
              betas=[[0.8, 0.0],      # cause 1: X0 matters
                     [0.0, -0.5]],    # cause 2: X1 matters
              model_cens="uniform", cens_par=50.0, max_time=50.0, seed=7)

df["status"].value_counts().sort_index()
```

`status` is the cause, so count it rather than summing it:

```text
0      716     # censored
1    12408     # cause 1
2     6876     # cause 2
```

## Check: does a cause-specific Cox model recover each row of `betas`?

Analyse one cause at a time, treating the other causes as censored — that is
what "cause-specific" means:

```python
from lifelines import CoxPHFitter

for cause in (1, 2):
    d = df.assign(event=(df["status"] == cause).astype(int))
    fit = CoxPHFitter().fit(d[["time", "event", "X0", "X1"]],
                            duration_col="time", event_col="event")
    print(f"cause {cause}: {fit.params_.round(3).to_dict()}")
```

```text
cause 1: {'X0': 0.8, 'X1': -0.002}
cause 2: {'X0': 0.014, 'X1': -0.513}
```

Both rows come back where they were set, and neither cause picks up the other's
covariate.

!!! note "Cause-specific hazards are not subdistribution hazards"

    A Fine-Gray model estimates something different — the effect on the
    cumulative incidence, keeping subjects who failed from other causes in the
    risk set. Its coefficients will **not** match the `betas` you passed here,
    and that is correct rather than a bug. `betas` are cause-specific by
    construction.

## Administrative censoring with `max_time`

`max_time` truncates follow-up for everybody, on top of the random censoring
from `model_cens`. Set it to `None` for no administrative cut-off, or to the
length of your hypothetical study to mimic one:

```python
df = generate(model="competing_risks", n=1000, n_risks=3,
              baseline_hazards=[0.3, 0.2, 0.1],
              max_time=2.0, cens_par=100.0, seed=1)
# nothing is observed after t = 2
```

## Related

- [Output schemas](../getting-started/schemas.md#competing-risks) — `status` as a cause code
- [Mixture cure](mixture-cure.md) — when some subjects never fail at all
- [Illness-death (CMM)](cmm.md) — when causes are states you can move between
- API: [`gen_competing_risks`, `gen_competing_risks_weibull`](../api/generators.md#gen_surv.competing_risks)
