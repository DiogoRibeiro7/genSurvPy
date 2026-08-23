# Piecewise exponential

`model="piecewise_exponential"` — a baseline hazard that is constant inside
intervals you choose and jumps between them. The cheapest way to get an
arbitrary hazard shape.

## The model

Given breakpoints $0 < \tau_1 < \dots < \tau_k$ and rates
$\lambda_0, \dots, \lambda_k$, the baseline hazard is

$$
h_0(t) = \lambda_j \quad \text{for } t \in [\tau_j, \tau_{j+1}),
$$

with $\tau_0 = 0$ and $\tau_{k+1} = \infty$. Covariates act proportionally:

$$
h(t \mid X) = h_0(t)\exp(X^\top\beta).
$$

**There is always one more rate than there are breakpoints** — two breakpoints
cut the timeline into three pieces.

## Parameters

```python
gen_piecewise_exponential(n, breakpoints, hazard_rates, betas=None,
                          n_covariates=2, covariate_dist="normal",
                          covariate_params=None, model_cens="uniform",
                          cens_par=5.0, seed=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | — | number of subjects |
| `breakpoints` | `list[float]` | — | increasing times where the hazard changes |
| `hazard_rates` | `list[float]` | — | one rate per interval — `len(breakpoints) + 1` of them |
| `betas` | `list[float]` \| `None` | `None` | coefficients; random values are drawn when omitted |
| `n_covariates` | `int` | `2` | number of covariates when `betas` is not given |
| `covariate_dist` | `"normal"` \| `"uniform"` \| `"binary"` | `"normal"` | covariate distribution — see [Covariates](../guides/covariates.md) |
| `covariate_params` | `dict` \| `None` | `None` | distribution parameters |
| `model_cens` | `str` | `"uniform"` | censoring mechanism |
| `cens_par` | `float` | `5.0` | censoring parameter |
| `seed` | `int` \| `Generator` \| `None` | `None` | reproducibility |

!!! tip "Pin `betas` when you are validating"

    With `betas=None` the coefficients are **drawn at random**, so you no
    longer know the truth you are testing against. Pass them explicitly —
    `betas=[0.0, 0.0]` isolates the baseline hazard entirely.

## Example

```python
from gen_surv import generate

df = generate(model="piecewise_exponential", n=6,
              breakpoints=[1.0, 3.0], hazard_rates=[0.5, 1.0, 0.2],
              seed=42)
print(df)
```

```text
 id     time  status        X0        X1
  0 3.790439       0  0.750451  0.940565
  1 1.772630       0 -1.951035 -1.302180
  2 0.980989       1  0.127840 -0.316243
  3 4.465606       0 -0.016801 -0.853044
  4 0.495887       1  0.879398  0.777792
```

## Check: does each interval have the hazard you asked for?

The occurrence/exposure estimate — events in an interval divided by the time
subjects spent at risk in it — should return each rate:

```python
import numpy as np
from gen_surv import generate

breakpoints, rates = [1.0, 3.0], [0.5, 2.0, 0.2]
df = generate(model="piecewise_exponential", n=40000,
              breakpoints=breakpoints, hazard_rates=rates,
              betas=[0.0, 0.0], model_cens="uniform", cens_par=100.0, seed=7)

edges = [0.0] + breakpoints + [float(df["time"].max())]
for lo, hi, true_h in zip(edges[:-1], edges[1:], rates):
    at_risk = df[df["time"] > lo]
    exposure = float(np.minimum(at_risk["time"], hi).sub(lo).clip(lower=0).sum())
    events = int(((df["time"] > lo) & (df["time"] <= hi) & (df["status"] == 1)).sum())
    print(f"[{lo}, {hi:.2f})  declared={true_h}  empirical={events / exposure:.3f}")
```

```text
[0.0, 1.00)   declared=0.5  empirical=0.501
[1.0, 3.00)   declared=2.0  empirical=2.011
[3.0, 37.31)  declared=0.2  empirical=0.202
```

Each interval carries the rate it was given. The open-ended last interval is
populated only by the survivors of the earlier ones, so it is the noisiest of
the three — widen `n` before reading much into it.

## Approximating a smooth hazard

Piecewise-constant hazards approximate any shape if you use enough pieces —
this is the standard trick behind piecewise-exponential survival models:

```python
import numpy as np

# Approximate a Weibull hazard, h(t) = 1.5 * t^0.5, on [0, 5]
grid = np.linspace(0, 5, 11)
breakpoints = list(grid[1:-1])
midpoints = (grid[:-1] + grid[1:]) / 2
hazard_rates = list(1.5 * midpoints ** 0.5)
```

Ten pieces reproduce a Weibull hazard closely enough for most purposes; the
approximation error is bounded by how much the true hazard moves within a
piece, so refine the grid where it is steepest rather than everywhere.

## Related

- [Cox proportional hazards](cphm.md) — constant baseline, no breakpoints
- [AFT models](aft.md) — monotone or unimodal hazards from a parametric family
- [Covariates](../guides/covariates.md) — `covariate_dist` and `covariate_params`
- API: [`gen_piecewise_exponential`](../api/generators.md#gen_surv.piecewise.gen_piecewise_exponential)
