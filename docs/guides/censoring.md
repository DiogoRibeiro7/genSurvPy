# Censoring

Every generator censors. Two mechanisms are built in, and a handful of extra
samplers are available if you want something else.

## The two built-in mechanisms

Each generator takes `model_cens` and `cens_par`:

| `model_cens` | Censoring time distribution | What `cens_par` is |
|---|---|---|
| `"uniform"` | $C \sim \mathrm{Uniform}(0, \texttt{cens\_par})$ | the upper bound |
| `"exponential"` | $C \sim \mathrm{Exponential}$ | the **mean** — not the rate |

A subject's row records $\min(T, C)$ and which one won. Censoring is
independent of the event time in both cases, so standard survival estimators
are unbiased — see [Cox PH](../models/cphm.md#check-does-a-cox-model-recover-beta).

!!! warning "`cens_par` is a mean for exponential censoring"

    Larger `cens_par` means **less** censoring for both mechanisms. If you are
    used to exponential parameterisations by rate, the direction here is the
    opposite of what you might expect.

## Hitting a target event rate

There is no closed-form knob for "censor 30% of subjects" — it depends on the
model's own hazard. Sweep and pick:

```python
from gen_surv import generate

for cens_par in (0.5, 1.0, 5.0, 100.0):
    u = generate(model="cphm", n=20000, beta=0.5, covariate_range=2.0,
                 model_cens="uniform", cens_par=cens_par, seed=1)
    e = generate(model="cphm", n=20000, beta=0.5, covariate_range=2.0,
                 model_cens="exponential", cens_par=cens_par, seed=1)
    print(f"cens_par={cens_par:>6}  uniform={u['status'].mean():.3f}  "
          f"exponential={e['status'].mean():.3f}")
```

```text
cens_par=   0.5  uniform=0.324  exponential=0.448
cens_par=   1.0  uniform=0.508  exponential=0.616
cens_par=   5.0  uniform=0.870  exponential=0.886
cens_par= 100.0  uniform=0.994  exponential=0.994
```

Those numbers are specific to this model and these parameters. Re-run the sweep
whenever you change the hazard.

### Switching censoring off

Set `cens_par` to something far beyond the largest event time:

```python
df = generate(model="cphm", n=10000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1e9, seed=42)
df["status"].mean()      # 1.0 — every event observed
```

Useful when you want to check a generator's distribution without censoring in
the way, as the model pages do.

### Administrative censoring

`competing_risks`, `competing_risks_weibull` and `mixture_cure` also take
`max_time`, which truncates follow-up for everyone on top of the random
censoring. That is the "the study ended" kind of censoring, as opposed to the
"this subject was lost" kind.

## The other samplers

`gen_surv` ships five censoring-time samplers and three class-based equivalents:

| Function | Signature | Distribution |
|---|---|---|
| [`runifcens`](../api/censoring.md#gen_surv.censoring.runifcens) | `(size, cens_par, rng=None)` | `Uniform(0, cens_par)` |
| [`rexpocens`](../api/censoring.md#gen_surv.censoring.rexpocens) | `(size, cens_par, rng=None)` | `Exponential(mean=cens_par)` |
| [`rweibcens`](../api/censoring.md#gen_surv.censoring.rweibcens) | `(size, scale, shape, rng=None)` | Weibull |
| [`rlognormcens`](../api/censoring.md#gen_surv.censoring.rlognormcens) | `(size, mean, sigma, rng=None)` | log-normal |
| [`rgammacens`](../api/censoring.md#gen_surv.censoring.rgammacens) | `(size, shape, scale, rng=None)` | Gamma |

| Class | Constructor | Call |
|---|---|---|
| [`WeibullCensoring`](../api/censoring.md#gen_surv.censoring.WeibullCensoring) | `(scale, shape)` | `instance(size, rng=None)` |
| [`LogNormalCensoring`](../api/censoring.md#gen_surv.censoring.LogNormalCensoring) | `(mean, sigma)` | `instance(size, rng=None)` |
| [`GammaCensoring`](../api/censoring.md#gen_surv.censoring.GammaCensoring) | `(shape, scale)` | `instance(size, rng=None)` |

!!! info "These are not wired into the generators"

    `model_cens` accepts only `"uniform"` and `"exponential"`; every generator
    resolves it against those two functions internally. The Weibull, log-normal
    and Gamma samplers are standalone — you apply them yourself, as below.
    Passing `model_cens="weibull"` raises:

    ```text
    ChoiceError: Argument 'model_cens' must be one of 'exponential', 'uniform';
    got 'weibull' of type str. Choose a valid option.
    ```

## Recipe: censor with your own distribution

Generate effectively uncensored data, then apply the censoring you want:

```python
import numpy as np
from gen_surv import generate, WeibullCensoring

# 1. No censoring: every `time` is a true event time
df = generate(model="cphm", n=10000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1e9, seed=42)

# 2. Your censoring distribution
rng = np.random.default_rng(0)
c = WeibullCensoring(scale=1.0, shape=1.5)(len(df), rng)

# 3. Apply it
censored = df.assign(
    time=np.minimum(df["time"], c),
    status=(df["time"] <= c).astype(float),
)

print(censored["status"].mean())     # 0.664
```

The same shape works with any of the samplers:

```python
from gen_surv import rgammacens

c = rgammacens(len(df), shape=2.0, scale=0.5, rng=rng)   # event rate 0.699
```

!!! danger "Do the minimum before you overwrite `time`"

    In the snippet above, `df["time"]` is still the uncensored event time when
    it is compared against `c`. If you assign the new `time` first and then
    compute `status`, every subject looks like an event. Build both columns in
    one `assign`, as shown.

### Dependent censoring

The recipe also lets you break the independence assumption on purpose — useful
for showing that an estimator relying on it goes wrong:

```python
# Censoring that arrives sooner for high-covariate subjects
c = rng.exponential(scale=1.0 / (0.5 + df["X0"]))
```

## Which mechanism should you pick?

- **`"uniform"`** gives a hard upper bound on follow-up. Everyone is out of the
  study by `cens_par`, which mimics a fixed-length trial.
- **`"exponential"`** has no upper bound and a long tail, closer to
  loss-to-follow-up that can happen at any time.

For heavy censoring, uniform is the harsher of the two at equal `cens_par`, as
the sweep above shows.

## Related

- [Output schemas](../getting-started/schemas.md) — how `status` encodes censoring per model
- [Summarising a dataset](summaries.md) — event and censoring counts
- API: [Censoring](../api/censoring.md)
