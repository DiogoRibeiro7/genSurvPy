# Reproducibility

Simulation results are worth nothing if you cannot regenerate them. Every
generator in `gen_surv` takes a `seed`.

## Pass a seed, get the same frame

```python
from gen_surv import generate

kwargs = dict(model="cphm", n=50, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0)

a = generate(seed=42, **kwargs)
b = generate(seed=42, **kwargs)

a.equals(b)      # True
```

Omit `seed` and you get fresh randomness on every call — fine for exploration,
wrong for anything you intend to report.

## Seeds can be integers or NumPy generators

Every generator accepts either:

=== "Integer seed"

    ```python
    df = generate(model="cphm", n=50, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=42)
    ```

=== "Generator object"

    ```python
    import numpy as np

    rng = np.random.default_rng(42)
    df = generate(model="cphm", n=50, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=rng)
    ```

Both produce the identical frame, because an integer seed is turned into
`numpy.random.default_rng(seed)` internally.

!!! warning "A shared generator advances between calls"

    Passing the *same* `Generator` object to two calls does **not** give you
    the same data twice — the generator's state moves on, which is usually what
    you want:

    ```python
    rng = np.random.default_rng(0)
    first  = generate(..., seed=rng)
    second = generate(..., seed=rng)   # different draws
    ```

    For two identical datasets, pass the same *integer* twice, or construct a
    fresh `default_rng(0)` for each call.

## Running a simulation study

Vary the seed, hold everything else fixed, and collect the estimates:

```python
import numpy as np
import pandas as pd
from gen_surv import generate
from lifelines import CoxPHFitter

estimates = []
for seed in range(200):
    df = generate(model="cphm", n=500, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=seed)
    fit = CoxPHFitter().fit(df, duration_col="time", event_col="status")
    estimates.append(float(fit.params_["X0"]))

estimates = np.array(estimates)
print(f"mean {estimates.mean():.3f}  sd {estimates.std(ddof=1):.3f}")
```

Sequential integers are perfectly good seeds for `default_rng`, which spreads
them across the state space — there is no need to pick "random-looking" ones.

For parallel work, spawn independent streams rather than reusing one generator
across processes:

```python
import numpy as np

parent = np.random.SeedSequence(20260823)
children = parent.spawn(8)          # 8 independent, reproducible streams

datasets = [generate(model="cphm", n=500, beta=0.5, covariate_range=2.0,
                     model_cens="uniform", cens_par=1.0,
                     seed=np.random.default_rng(child))
            for child in children]
```

## What stability you can rely on

| | Guaranteed? |
|---|---|
| Same seed, same version, same parameters → identical frame | **Yes** |
| Same seed across operating systems and CPUs | **Yes** — NumPy's PCG64 is platform-independent |
| Same seed across `gen_surv` versions | **No** |

A bug fix in a sampler changes the draws it makes, so a patch release can
change the data a given seed produces. That has already happened: the 1.3.0 and
2.0.0 releases corrected sampling bugs and deliberately changed output, and
3.0.0 rebuilt `cmm` and `thmm` on the
[multistate engine](../models/multistate.md), which draws one candidate per
outgoing edge per visit where the old implementations drew all three latent
times up front. Those two generators changed for every seed.

**If a result must be reproducible for a paper, pin the version:**

```
gen-surv==3.1.0
```

and record it alongside the seed. Record both in the artefact itself where you
can:

```python
import gen_surv

df.attrs["gen_surv_version"] = gen_surv.__version__
df.attrs["seed"] = 42
```

## Reproducibility is not the same as correctness

A seeded run reproduces whatever the generator does, including anything it does
wrong. When you rely on a model's exact distributional properties, check them:
simulate a large sample and confirm the quantity you care about — a hazard
ratio, an interval hazard, a cure fraction — comes back where you set it. Each
model page shows this check for that model, and
[Summarising a dataset](../guides/summaries.md) covers the general tooling.
