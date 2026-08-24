# Summarising a dataset

Three functions for looking at a generated frame before you model it: a quick
table, a structured summary, and a quality check.

## The quick look

[`describe_survival`](../api/analysis.md#gen_surv.visualization.describe_survival)
returns a small two-column frame, meant for printing:

```python
from gen_surv import generate, describe_survival

df = generate(model="cphm", n=2000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)

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

The median is the Kaplan-Meier median, so it accounts for censoring — unlike
`df["time"].median()`, which does not.

## The structured summary

[`summarize_survival_dataset`](../api/analysis.md#gen_surv.summary.summarize_survival_dataset)
returns a nested dict, and prints a formatted report unless you pass
`verbose=False`:

```python
from gen_surv.summary import summarize_survival_dataset

summary = summarize_survival_dataset(df, verbose=False)
summary.keys()
```

```text
dict_keys(['dataset_info', 'event_info', 'time_info', 'data_quality', 'covariates'])
```

| Key | Contains |
|---|---|
| `dataset_info` | `n_subjects`, `n_unique_ids`, `n_covariates` |
| `event_info` | `n_events`, `n_censored`, `event_rate` |
| `time_info` | `min`, `max`, `mean`, `median` of observed time |
| `data_quality` | missing and invalid counts, plus an `overall_quality` verdict |
| `covariates` | per column: type, min, max, mean, median, std, missing, unique values |

```python
summary["event_info"]
# {'n_events': 1007.0, 'n_censored': 993.0, 'event_rate': 0.5035}

summary["data_quality"]["overall_quality"]
# 'good'
```

Because it is a plain dict, it goes straight into a results table:

```python
import pandas as pd

rows = []
for cens_par in (0.5, 1.0, 5.0):
    d = generate(model="cphm", n=5000, beta=0.5, covariate_range=2.0,
                 model_cens="uniform", cens_par=cens_par, seed=1)
    s = summarize_survival_dataset(d, verbose=False)
    rows.append({"cens_par": cens_par, **s["event_info"], **s["time_info"]})

pd.DataFrame(rows)
```

### Non-default column names

All three functions take `time_col` and `status_col`, which matters for the
models that do not use those names:

```python
from gen_surv import generate

tdcm_df = generate(model="tdcm", n=200, dist="weibull", corr=0.5,
                   dist_par=[1.0, 2.0, 1.0, 2.0], model_cens="uniform",
                   cens_par=5.0, beta=[0.5, 0.3], lam=1.0, seed=1)

# tdcm calls its time column `stop`
summarize_survival_dataset(tdcm_df, time_col="stop", verbose=False)
```

`thmm` output has neither a `time`-plus-`status` pair nor an event indicator at
all, so these helpers do not apply to it directly — derive an indicator from the
last state first, as shown in [THMM](../models/thmm.md).

## The quality check

[`check_survival_data_quality`](../api/analysis.md#gen_surv.summary.check_survival_data_quality)
returns `(frame, report)` and can repair problems rather than only reporting
them:

```python
from gen_surv.summary import check_survival_data_quality

clean, report = check_survival_data_quality(df)
report
```

```text
{'missing_data':   {'time': 0, 'status': 0, 'id': None},
 'invalid_values': {'negative_time': 0, 'excessive_time': 0, 'invalid_status': 0},
 'duplicates':     {'duplicate_rows': 0, 'duplicate_ids': None},
 'modifications':  {'rows_dropped': 0, 'values_fixed': 0}}
```

| Parameter | Default | Effect |
|---|---|---|
| `min_time` | `0.0` | times below this count as invalid |
| `max_time` | `None` | times above this count as `excessive_time` |
| `status_values` | `None` | the set of allowed status codes — pass `[0, 1, 2]` for competing risks |
| `fix_issues` | `False` | when `True`, drop or repair the offending rows and record it under `modifications` |
| `id_col` | `None` | enables the duplicate-id check |

Freshly generated data is clean by construction, so this is mostly for data you
have modified — after custom censoring, a merge, or a round trip through CSV:

```python
from gen_surv import generate

reloaded = generate(model="aft_ln", n=500, beta=[0.5], sigma=1.0,
                    model_cens="uniform", cens_par=2.0, seed=1)

clean, report = check_survival_data_quality(
    reloaded, id_col="id", max_time=10.0, status_values=[0, 1], fix_issues=True
)
print(report["modifications"])
```

## Comparing datasets

[`compare_survival_datasets`](../api/analysis.md#gen_surv.summary.compare_survival_datasets)
takes a dict of frames and returns one row per dataset — the fastest way to see
what a parameter sweep did:

```python
from gen_surv import generate
from gen_surv.summary import compare_survival_datasets

datasets = {
    f"beta={b}": generate(model="cphm", n=2000, beta=b, covariate_range=2.0,
                          model_cens="uniform", cens_par=1.0, seed=1)
    for b in (0.0, 0.5, 1.0)
}

compare_survival_datasets(datasets)
```

Pair it with [`plot_hazard_comparison`](plotting.md) to see the same comparison
as curves.

## Related

- [Plotting](plotting.md)
- [Output schemas](../getting-started/schemas.md)
- API: [Analysis](../api/analysis.md)
