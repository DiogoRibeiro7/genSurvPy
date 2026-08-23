# Fitting models to the data

Generated frames are ordinary pandas objects, so most survival libraries take
them as they are. Two of them need a conversion, and `gen_surv` provides it.

## lifelines — nothing to convert

[lifelines](https://lifelines.readthedocs.io) works on DataFrames directly, and
is a hard dependency, so it is already installed:

```python
from gen_surv import generate
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter

df = generate(model="cphm", n=5000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=7)

CoxPHFitter().fit(df, duration_col="time", event_col="status").print_summary()

KaplanMeierFitter().fit(df["time"], df["status"])

WeibullAFTFitter().fit(df, duration_col="time", event_col="status")
```

Pass only the columns the model should see — `CoxPHFitter` treats every
remaining column as a covariate, so an `id` column would be fitted as one:

```python
cph = CoxPHFitter().fit(df[["time", "status", "X0", "X1"]],
                        duration_col="time", event_col="status")
```

## scikit-survival — a structured array

[scikit-survival](https://scikit-survival.readthedocs.io) wants a NumPy
structured array of `(event, time)` pairs.
[`to_sksurv`](../api/interoperability.md#gen_surv.integration.to_sksurv) builds
one:

```python
from gen_surv import generate, to_sksurv

df = generate(model="cphm", n=200, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)

y = to_sksurv(df)
y.dtype       # dtype([('status', '?'), ('time', '<f8')])
y[:3]         # [(False, 0.43887844) (False, 0.09417735) (True, 0.03704099)]
```

Note the conversion: `status` becomes **boolean**, which is what
scikit-survival expects.

```python
from sksurv.linear_model import CoxPHSurvivalAnalysis

X = df[["X0"]].to_numpy()
model = CoxPHSurvivalAnalysis().fit(X, y)
model.coef_
```

Going back the other way:

```python
from gen_surv import from_sksurv

from_sksurv(y).columns      # ['time', 'status']
```

`from_sksurv` returns only the time and status columns — the covariates were
never in `y` to begin with, so rejoin them yourself if you need the full frame.

!!! info "scikit-survival is the one optional dependency"

    `pip install scikit-survival`. Without it, `to_sksurv` and `from_sksurv`
    are not importable from `gen_surv`; everything else works. See
    [Installation](../getting-started/installation.md#the-one-optional-extra).

Both functions take `time_col` and `event_col`, so non-default names are fine:

```python
y = to_sksurv(tdcm_df, time_col="stop", event_col="status")
```

## scikit-learn — a generator as an estimator

[`GenSurvDataGenerator`](../api/interoperability.md#gen_surv.sklearn_adapter.GenSurvDataGenerator)
wraps `generate()` in the estimator interface, so a dataset can be produced
inside a pipeline or swept with the usual scikit-learn tooling:

```python
from gen_surv import GenSurvDataGenerator

est = GenSurvDataGenerator("cphm", n=500, beta=0.5, covariate_range=1.0,
                           model_cens="uniform", cens_par=1.0, seed=1)

df = est.fit_transform()
list(df.columns)      # ['time', 'status', 'X0']
```

| Argument | Meaning |
|---|---|
| first positional | the model name, exactly as for `generate()` |
| `return_type` | `"df"` (default) for a DataFrame, `"dict"` for a plain dict of columns |
| everything else | forwarded to `generate()` unchanged |

```python
est = GenSurvDataGenerator("cphm", return_type="dict", n=5, beta=0.5,
                           covariate_range=1.0, model_cens="uniform",
                           cens_par=1.0, seed=1)
est.fit_transform().keys()      # dict_keys(['time', 'status', 'X0'])
```

`fit` is a no-op that returns `self`; the data is produced by `transform`. It
implements `get_params` and `set_params`, so parameters can be varied the
scikit-learn way:

```python
est.set_params(n=1000)
```

An invalid `return_type` is rejected at construction, not at transform time.

## R — through RDS

```python
from gen_surv import export_dataset

export_dataset(df, "survival.rds")
```

```r
df <- readRDS("survival.rds")
survival::coxph(Surv(time, status) ~ X0, data = df)
```

The natural way to check a Python result against the original R
[genSurv](https://cran.r-project.org/package=genSurv) package. See
[Exporting data](export.md).

## Multi-state frames need different tools

`cmm` and `thmm` are not one-row-per-subject, so `CoxPHFitter` and
`to_sksurv` do not apply to them unchanged:

- **`cmm`** is already in `(start, stop]` form. Fit one transition at a time,
  or use a stratified model — see [CMM](../models/cmm.md#fitting-a-transition-specific-model).
- **`thmm`** is a state panel. The natural target is a multi-state Markov
  likelihood, such as R's `msm` package.

Trying either on a multi-state frame silently fits the wrong thing rather than
raising, because the columns are all numeric. Check
[Output schemas](../getting-started/schemas.md) before pointing an estimator at
a frame you did not generate yourself.

## Related

- [Exporting data](export.md)
- [Output schemas](../getting-started/schemas.md)
- API: [Interoperability](../api/interoperability.md)
