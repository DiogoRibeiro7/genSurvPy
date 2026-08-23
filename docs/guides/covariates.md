# Covariates

Different model families build their covariates in different ways. This page
maps out which is which, and how to control them.

## Three schemes

| Scheme | Models | How covariates are drawn |
|---|---|---|
| **Range** | `cphm`, `cmm`, `thmm` | one covariate `X0` from `Uniform(0, covariate_range)` |
| **Standard normal** | `aft_ln`, `aft_weibull`, `aft_log_logistic` | `len(beta)` covariates, each `Normal(0, 1)` — not configurable |
| **Configurable** | `piecewise_exponential`, `competing_risks`, `competing_risks_weibull`, `mixture_cure` | `covariate_dist` and `covariate_params` |

`tdcm` is its own case: its baseline covariate comes from a bivariate draw
shared with the crossover time — see [that model's page](../models/tdcm.md).

## Range-based models

One knob, one covariate:

```python
from gen_surv import generate

df = generate(model="cphm", n=1000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=1)
df["X0"].min(), df["X0"].max()     # ~0.0, ~2.0
```

`covariate_range` must be positive. Widening it widens the spread of the linear
predictor, which makes the covariate's effect easier to estimate — a useful dial
when you are studying power.

## Configurable models

```python
generate(model="...", covariate_dist="normal", covariate_params=None, ...)
```

| `covariate_dist` | Required keys in `covariate_params` | Default when `None` |
|---|---|---|
| `"normal"` | `mean`, `std` | `{"mean": 0.0, "std": 1.0}` |
| `"uniform"` | `low`, `high` | `{"low": 0.0, "high": 1.0}` |
| `"binary"` | `p` | `{"p": 0.5}` |

All covariates in a dataset share the distribution — there is no per-covariate
specification.

```python
from gen_surv import generate

for dist, params in [("normal", None),
                     ("normal", {"mean": 10.0, "std": 2.0}),
                     ("uniform", {"low": -1.0, "high": 1.0}),
                     ("binary", {"p": 0.3})]:
    d = generate(model="piecewise_exponential", n=5000, breakpoints=[1.0],
                 hazard_rates=[0.5, 0.5], betas=[0.0, 0.0],
                 covariate_dist=dist, covariate_params=params, seed=1)
    print(f"{dist:8} {str(params):30} mean={d['X0'].mean():7.3f} "
          f"min={d['X0'].min():7.3f} max={d['X0'].max():7.3f}")
```

```text
normal   None                           mean= -0.010 min= -3.838 max=  3.932
normal   {'mean': 10.0, 'std': 2.0}     mean=  9.981 min=  2.324 max= 17.864
uniform  {'low': -1.0, 'high': 1.0}     mean= -0.004 min= -1.000 max=  0.999
binary   {'p': 0.3}                     mean=  0.296 min=  0.000 max=  1.000
```

`"binary"` gives a 0/1 covariate — the natural choice for a treatment arm.

### Partial parameter dicts are rejected

Supply a `covariate_params` and you must supply every key for that
distribution. There is no per-key defaulting:

```python
generate(model="piecewise_exponential", n=10, breakpoints=[1.0],
         hazard_rates=[0.5, 0.5], covariate_dist="normal",
         covariate_params={"mean": 0.0})
```

```text
ParameterError: Invalid value for 'covariate_params': {'mean': 0.0} (type dict).
must include 'mean' and 'std'. Check and adjust this parameter.
(while validating inputs for model 'piecewise_exponential')
```

## How many covariates do you get?

| Model | Count comes from |
|---|---|
| `cphm`, `cmm`, `thmm` | always exactly one, `X0` |
| `aft_*` | `len(beta)` |
| `piecewise_exponential`, `mixture_cure` | `len(betas)` if given, otherwise `n_covariates` (default 2) |
| `competing_risks*` | the width of the `betas` matrix if given, otherwise `n_covariates` |
| `tdcm` | one baseline covariate plus the `tdcov` indicator |

## Always pin your coefficients when validating

For the configurable models, leaving `betas` at `None` makes the package
**draw the coefficients at random**:

```python
# Coefficients are random — you no longer know the truth
df = generate(model="piecewise_exponential", n=1000, breakpoints=[1.0],
              hazard_rates=[0.5, 1.0], seed=1)

# Coefficients are yours
df = generate(model="piecewise_exponential", n=1000, breakpoints=[1.0],
              hazard_rates=[0.5, 1.0], betas=[0.4, -0.2], seed=1)
```

This is convenient for a smoke test and fatal for a simulation study. If your
estimator is being compared against a number, that number has to be one you
chose.

To study the baseline hazard on its own, zero them out:

```python
betas=[0.0, 0.0]      # covariates present, but with no effect
```

## Adding covariates the package cannot make

The frames are ordinary pandas objects, so derive whatever you need:

```python
import numpy as np

rng = np.random.default_rng(0)
df["age"] = rng.normal(65, 10, len(df))
df["arm"] = rng.binomial(1, 0.5, len(df))
```

Anything added this way has **no effect on the generated times** — the event
times were drawn before you attached the column. To let a covariate matter, it
has to be one the generator knows about, through `betas`.

## Related

- [Choosing a model](../models/index.md)
- [Output schemas](../getting-started/schemas.md) — covariate naming per model
- API: [Generators](../api/generators.md)
