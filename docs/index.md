# gen_surv

**Simulate survival data with a known truth.** `gen_surv` generates synthetic
time-to-event datasets from twelve models — proportional hazards, accelerated
failure time, competing risks, cure fractions, piecewise hazards, recurrent
events and two illness-death processes — so you can test an estimator against
parameters you chose yourself.

It is a Python port of the R package
[genSurv](https://cran.r-project.org/package=genSurv), extended well past the
original's four models.

[Install it](getting-started/installation.md){ .md-button .md-button--primary }
[Generate your first dataset](getting-started/quickstart.md){ .md-button }

---

## Thirty seconds

```python
from gen_surv import generate

df = generate(
    model="cphm",           # Cox proportional hazards
    n=6,
    beta=0.5,               # log hazard ratio
    covariate_range=2.0,    # X0 ~ Uniform(0, 2)
    model_cens="uniform",
    cens_par=1.0,
    seed=42,
)
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

You picked `beta = 0.5`, so you know what a correct estimator should recover.
That is the whole point: every column above was produced by a mechanism you
specified.

## What it is for

<div class="grid cards" markdown>

-   :material-flask: **Method development**

    Check that a new estimator recovers the parameters that generated the data,
    across sample sizes and censoring levels.

-   :material-school: **Teaching**

    Hand students a dataset whose hazard ratio, cure fraction or transition
    intensities are known, and let them try to find them.

-   :material-speedometer: **Benchmarking**

    Compare implementations on identical inputs — every generator takes a
    `seed`, so runs are byte-for-byte reproducible.

-   :material-check-decagram: **Software validation**

    Exercise an analysis pipeline against edge cases: heavy censoring, a cured
    subpopulation, non-proportional hazards, panel-observed states.

</div>

## The twelve models

| `model=` | Family | Returns | Page |
|---|---|---|---|
| `cphm` | Cox proportional hazards | one row per subject | [Cox PH](models/cphm.md) |
| `aft_ln` | Log-normal AFT | one row per subject | [AFT](models/aft.md) |
| `aft_weibull` | Weibull AFT | one row per subject | [AFT](models/aft.md) |
| `aft_log_logistic` | Log-logistic AFT | one row per subject | [AFT](models/aft.md) |
| `piecewise_exponential` | Piecewise constant hazard | one row per subject | [Piecewise](models/piecewise-exponential.md) |
| `competing_risks` | Cause-specific constant hazards | one row per subject | [Competing risks](models/competing-risks.md) |
| `competing_risks_weibull` | Cause-specific Weibull hazards | one row per subject | [Competing risks](models/competing-risks.md) |
| `mixture_cure` | Logistic cure + exponential failure | one row per subject | [Mixture cure](models/mixture-cure.md) |
| `cmm` | Illness-death, counting-process intervals | two or three rows per subject | [CMM](models/cmm.md) |
| `thmm` | Illness-death, observed state panel | two or three rows per subject | [THMM](models/thmm.md) |
| `tdcm` | Cox with a time-dependent covariate | one row per subject | [TDCM](models/tdcm.md) |
| `recurrent_events` | Repeated events per subject (AG, PWP) | one row per at-risk interval | [Recurrent events](models/recurrent-events.md) |

There is a thirteenth generator that `generate()` cannot reach.
[`gen_multistate`](models/multistate.md) takes an arbitrary transition graph as
a list of objects rather than a set of scalars, so it has no `model=` string and
no command-line form. `cmm` and `thmm` are configurations of it.

Not sure which one you need? [Choosing a model](models/index.md) walks through
the decision.

!!! warning "The output shape is not the same for every model"

    Multi-state generators return several rows per subject, and column names
    differ between families — `cphm` has no `id` column at all, `tdcm` uses
    `covariate` where the others use `X0`. Read
    [Output schemas](getting-started/schemas.md) before you write code that
    consumes a generated frame.

## The ground truth, not just the data

A generated frame looks like a real one, which means it hides the same things.
[`simulate()`](guides/simulation-results.md) hands back what a real dataset
never could — the coefficients actually used, the event time before censoring
intervened, which subjects are cured, when a covariate crossed over:

```python
from gen_surv import simulate

result = simulate("cphm", n=1000, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=42)

result.truth["event_time"]       # when each subject would have failed
result.truth["censoring_time"]   # what censoring hid
```

Several models draw their coefficients for you when you omit them, and
`result.truth["betas"]` is the only way to learn what they were.

## Beyond generating

- [Ground truth](guides/simulation-results.md) — configurations, latent times,
  and the coefficients a model chose for itself.
- [Baseline hazards](guides/baselines.md) — five families, and the protocol for
  writing your own.
- [Censoring](guides/censoring.md) — the two mechanisms wired into every
  generator, and the standalone samplers for everything else.
- [Summarising a dataset](guides/summaries.md) — event counts, follow-up,
  quality checks, dataset comparison.
- [Plotting](guides/plotting.md) — Kaplan-Meier curves, hazard comparisons,
  covariate effects.
- [Fitting models to the data](guides/interoperability.md) — handing the frame
  to lifelines, scikit-survival or scikit-learn.
- [Command line](guides/cli.md) — generate and plot without writing Python.

## Citing

```bibtex
@software{ribeiro_gensurv,
  title   = {gen_surv: Survival Data Simulation in Python},
  author  = {Diogo Ribeiro},
  url     = {https://github.com/DiogoRibeiro7/genSurvPy},
  version = {3.1.2}
}
```

Machine-readable metadata lives in
[`CITATION.cff`](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/CITATION.cff)
and [`.zenodo.json`](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/.zenodo.json).
