# Exporting data

[`export_dataset`](../api/interoperability.md#gen_surv.export.export_dataset)
writes a generated frame to disk in four formats, inferring the format from the
file extension.

```python
from gen_surv import export_dataset, generate

df = generate(model="cphm", n=1000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)

export_dataset(df, "survival.csv")
```

## Formats

| Extension | `fmt=` | Written with | Use it for |
|---|---|---|---|
| `.csv` | `"csv"` | `DataFrame.to_csv(index=False)` | anything, everywhere |
| `.json` | `"json"` | `DataFrame.to_json(orient="table")` | round trips that must keep dtypes |
| `.feather`, `.ft` | `"feather"`, `"ft"` | `DataFrame.to_feather` | large frames, fast reload |
| `.rds` | `"rds"` | `pyreadr.write_rds` | handing data to R |

The extension decides unless you say otherwise:

```python
export_dataset(df, "survival.dat", fmt="csv")     # extension ignored
```

An unsupported format fails before writing anything:

```text
ChoiceError: Argument 'fmt' must be one of 'csv', 'feather', 'ft', 'json',
'rds'; got 'parquet' of type str. Choose a valid option.
```

!!! tip "Parquet is not in the list, but pandas is right there"

    ```python
    df.to_parquet("survival.parquet")     # pyarrow is already a dependency
    ```

    `export_dataset` is a convenience wrapper, not a restriction — the frame is
    an ordinary `DataFrame` and every pandas writer works on it.

## Handing data to R

RDS is the reason `pyreadr` is a dependency:

```python
export_dataset(df, "survival.rds")
```

```r
df <- readRDS("survival.rds")
library(survival)
coxph(Surv(time, status) ~ X0, data = df)
```

This is the natural way to compare against the original R
[genSurv](https://cran.r-project.org/package=genSurv) package, or to check a
Python estimator against `survival::coxph`.

## Round trips lose things

CSV keeps no dtype information, and the multi-state frames are where that
bites:

```python
import pandas as pd
from gen_surv import generate

df = generate(model="cmm", n=100, model_cens="uniform", cens_par=2.0,
              beta=[0.1, 0.2, 0.3], covariate_range=1.0,
              rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0], seed=1)

df.to_csv("cmm.csv", index=False)
back = pd.read_csv("cmm.csv")

df.dtypes.equals(back.dtypes)     # may be False
```

For anything where dtypes matter, use Feather or the `"json"` format, which
writes a schema alongside the data:

```python
export_dataset(df, "cmm.feather")
back = pd.read_feather("cmm.feather")      # dtypes preserved
```

## Record what produced the file

A dataset without its parameters is not reproducible. Write them next to it:

```python
import json
from pathlib import Path
import gen_surv

params = dict(model="cphm", n=1000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)

df = gen_surv.generate(**params)
gen_surv.export_dataset(df, "survival.csv")

Path("survival.meta.json").write_text(json.dumps(
    {**params, "gen_surv_version": gen_surv.__version__}, indent=2
))
```

The version matters as much as the seed — a patch release can change what a
given seed produces. See
[Reproducibility](../getting-started/reproducibility.md).

## From the command line

```bash
gen_surv dataset cphm --n 1000 --beta 0.5 --seed 42 -o survival.csv
```

The CLI writes CSV only; use Python for the other formats. See
[Command line](cli.md).

## Related

- [Fitting models to the data](interoperability.md) — passing frames to other libraries in memory
- API: [Interoperability](../api/interoperability.md)
