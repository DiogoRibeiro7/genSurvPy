# Quickstart

This page takes one dataset from generation to a fitted model. Every number
shown is real output from the code above it.

## 1. Generate

`generate()` is the single entry point. You pass a model name and that model's
parameters:

```python
from gen_surv import generate

df = generate(
    model="cphm",
    n=2000,
    beta=0.5,
    covariate_range=2.0,
    model_cens="uniform",
    cens_par=1.0,
    seed=42,
)
df.head()
```

```text
       time  status        X0
0  0.438878     0.0  1.547912
1  0.094177     0.0  1.394736
2  0.037041     1.0  1.522279
3  0.370798     0.0  0.900772
4  0.646901     1.0  1.287730
```

Three columns, one row per subject:

| Column | Meaning |
|---|---|
| `time` | the observed time — the event time, or the censoring time if that came first |
| `status` | `1.0` if the event was observed, `0.0` if the subject was censored |
| `X0` | the covariate, drawn from `Uniform(0, covariate_range)` |

The mechanism behind those rows: for subject $i$, draw $X_i \sim
\mathrm{Uniform}(0, 2)$, draw an event time $T_i \sim \mathrm{Exponential}$ with
rate $\exp(\beta X_i)$, draw a censoring time $C_i \sim \mathrm{Uniform}(0, 1)$,
and report $\min(T_i, C_i)$ along with which one won.

## 2. Look at what you got

```python
from gen_surv import describe_survival

describe_survival(df)
```

```text
              Metric   Value
  Total Observations    2000
    Number of Events  1007.0
     Number Censored   993.0
          Event Rate  50.35%
Median Survival Time  0.4065
            Min Time  0.0001
            Max Time  0.9994
            Mean Time  0.3003
```

Roughly half the subjects are censored, which is what `cens_par=1.0` buys with
these hazards. Turn that dial and the event rate moves — see
[Censoring](../guides/censoring.md).

## 3. Recover the parameter you chose

This is the point of simulated data. You set `beta=0.5`; a correct estimator
should find it.

```python
from gen_surv import generate
from lifelines import CoxPHFitter

df = generate(model="cphm", n=5000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=7)

cph = CoxPHFitter().fit(df, duration_col="time", event_col="status")
cph.summary[["coef", "se(coef)", "coef lower 95%", "coef upper 95%"]]
```

```text
            coef  se(coef)  coef lower 95%  coef upper 95%
covariate
X0         0.501     0.035           0.433           0.569
```

0.501, against a truth of 0.5.

!!! warning "Use enough subjects before you conclude anything"

    At `n=2000` the same fit gives 0.383 with a 95% interval of
    [0.275, 0.491] — an interval that happens to exclude the true value. That
    is ordinary sampling variation, not a broken generator. When you are
    validating an estimator, either use a large `n` or repeat over many seeds
    and look at the distribution of estimates.

## 4. Plot it

```python
import matplotlib.pyplot as plt
from gen_surv import plot_survival_curve

fig, ax = plot_survival_curve(df, title="Kaplan-Meier, CPHM sample")
plt.show()
```

To see the covariate's effect, split it into groups:

```python
from gen_surv import plot_covariate_effect

fig, ax = plot_covariate_effect(df, covariate_col="X0", n_groups=3)
```

More in [Plotting](../guides/plotting.md).

## 5. Save it

```python
from gen_surv import export_dataset

export_dataset(df, "cphm.csv")       # format inferred from the extension
export_dataset(df, "cphm.rds")       # for R
```

See [Exporting data](../guides/export.md) for the four supported formats.

## Doing it without Python

The same dataset from the shell:

```bash
gen_surv dataset cphm --n 2000 --beta 0.5 --covariate-range 2.0 \
    --model-cens uniform --cens-par 1.0 --seed 42 -o cphm.csv
```

See [Command line](../guides/cli.md).

## Where next

- **A different model?** [Choosing a model](../models/index.md) maps research
  questions onto the eleven generators.
- **Consuming the output in code?** [Output schemas](../getting-started/schemas.md)
  documents every column of every model, including where they disagree.
- **Need runs to be identical?** [Reproducibility](reproducibility.md).
