# Configuration and ground truth

`generate()` returns a DataFrame — what an analyst wants, and what an estimator
consumes. It is not what a **methodologist** wants. The interesting quantities
in a simulation study are precisely the ones a real dataset could never
contain: the coefficients that produced it, the event time before censoring
intervened, which subjects are cured, when a covariate crossed over.

`simulate()` returns those alongside the frame.

```python
from gen_surv import simulate

result = simulate("cphm", n=100, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=42)

result.data              # exactly what generate() returns
result.config            # the call that produced it
result.truth             # what the frame cannot show
```

```text
SimulationResult(model='cphm', rows=100, subjects=100,
                 truth=['beta', 'censoring_time', 'covariates',
                        'event_time', 'linear_predictor'])
```

`generate()` and the `gen_*` functions are unchanged. This is an addition.

## What the truth is good for

### Recovering coefficients you never chose

Several generators **draw their coefficients at random** when you omit them.
Until now there was no way to find out what they were, which quietly made those
datasets useless for validating an estimator:

```python
result = simulate("piecewise_exponential", n=100,
                  breakpoints=[1.0], hazard_rates=[0.5, 1.5], seed=42)

result.truth["betas"]          # array([0.0006, 0.1494])
```

### Seeing what censoring hid

```python
result = simulate("cphm", n=1000, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=1)

event = result.truth["event_time"]
censoring = result.truth["censoring_time"]

(event > censoring).mean()          # the true censoring rate
event[event > censoring].mean()     # when those subjects would have failed
```

The observed `time` column is `min(event, censoring)` by construction, so these
two arrays reconstruct it exactly — which is the check that the truth report
describes the data it came with.

### Building the analysis dataset a naive frame cannot support

`tdcm` records only the value of its time-dependent covariate at exit, never
the moment it switched. The crossover time is in the truth:

```python
result = simulate("tdcm", n=500, dist="weibull", corr=0.5,
                  dist_par=[1.0, 2.0, 1.0, 2.0], model_cens="uniform",
                  cens_par=5.0, beta=[0.5, 0.3], lam=1.0, seed=1)

result.truth["crossover_time"]      # one per subject
```

With it you can split each subject at the crossover into the two risk intervals
a time-varying Cox model needs.

## The keys

Shared vocabulary, where a model has the quantity:

| Key | Meaning |
|---|---|
| `beta` / `betas` | the coefficients actually used |
| `covariates` | the covariate matrix |
| `linear_predictor` | `covariates @ betas` |
| `event_time` | the latent event time, before censoring was applied |
| `censoring_time` | the drawn censoring time |

Model-specific:

| Model | Additional keys |
|---|---|
| `piecewise_exponential` | `breakpoints`, `hazard_rates` |
| `competing_risks`, `competing_risks_weibull` | `cause_times` (one column per cause), `first_event_time` |
| `mixture_cure` | `betas_survival`, `betas_cure`, `cure_linear_predictor`, `cured` |
| `cmm`, `thmm` | `rate`, `transition_times` (`t12`, `t13`, `t23`) |
| `tdcm` | `crossover_time`, `switched_before_exit` |
| `recurrent_events` | `baseline` (the object), `followup_end`, `dropout_time` |

## As a frame

`truth_frame()` returns the per-subject entries, ready to join onto the data:

```python
result = simulate("mixture_cure", n=500, cure_fraction=0.3, seed=1)
result.truth_frame().head()
```

Scalars and anything not one-per-subject are left out, so the result always
lines up with the subjects.

## Configurations as values

`SimulationConfig` is the specification: model, parameters, and the `gen_surv`
version that ran it. Version matters — a bug fix in a sampler changes what a
seed produces, so a configuration without one is not reproducible.

```python
from gen_surv import SimulationConfig

base = SimulationConfig("cphm", {"n": 500, "beta": 0.5, "covariate_range": 2.0,
                                 "model_cens": "uniform", "cens_par": 1.0})

for seed in range(200):
    result = base.replace(seed=seed).run()
    ...
```

`replace()` returns a copy, so `base` is never mutated and a sweep reads as one
thing varying.

Configurations serialise, which is what makes a scenario shareable:

```python
import json

text = json.dumps(base.to_dict())
restored = SimulationConfig.from_dict(json.loads(text))

restored == base      # True
```

## How it works

Generators do not change signature and do not return tuples. Each calls an
internal `record()` at the point where the values exist; outside a capture
block that is a no-op, so `gen_*` behaves exactly as before. `simulate()` opens
the block and collects what lands in it.

The sink is a `ContextVar`, so concurrent simulations in threads or async tasks
cannot see each other's truth.

This design was chosen so that adding the feature could not perturb a single
draw: the [frozen output baselines](https://github.com/DiogoRibeiro7/genSurvPy/tree/develop/tests/baselines)
for all twelve generators are byte-identical before and after.

## Related

- [Reproducibility](../getting-started/reproducibility.md) — seeds, versions and what is guaranteed
- [Choosing a model](../models/index.md)
- API: [Simulation results](../api/results.md)
