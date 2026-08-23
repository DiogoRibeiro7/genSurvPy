# Command line

Installing the package puts a `gen_surv` command on your path. It has two
subcommands.

```bash
gen_surv --help
```

```text
 Usage: gen_surv [OPTIONS] COMMAND [ARGS]...

 Generate synthetic survival datasets.

+- Commands ------------------------------------------------------------------+
| dataset     Generate survival data and optionally save to CSV.              |
| visualize   Visualize survival data from a CSV file.                        |
+-----------------------------------------------------------------------------+
```

`python -m gen_surv` does the same thing, which is handy when the script
directory is not on `PATH`.

## `dataset`

```bash
gen_surv dataset MODEL [OPTIONS]
```

`MODEL` is any of the twelve names — `cphm`, `cmm`, `tdcm`, `thmm`, `aft_ln`,
`aft_weibull`, `aft_log_logistic`, `competing_risks`,
`competing_risks_weibull`, `mixture_cure`, `piecewise_exponential`,
`recurrent_events`.

Writes CSV to the path given by `-o`, or to stdout when it is omitted:

```bash
# To a file
gen_surv dataset cphm --n 1000 --beta 0.5 --covariate-range 2.0 -o cphm.csv

# To stdout, so it pipes
gen_surv dataset aft_ln --n 500 --beta 0.5 --beta -0.3 --sigma 1.0 | head -5
```

### Options

| Option | Default | Applies to |
|---|---|---|
| `--n` | `100` | all models |
| `--model-cens` | `uniform` | all models |
| `--cens-par` | `1.0` | all models |
| `--beta` | `0.5` | all models — **repeat the flag** for several coefficients |
| `--covariate-range`, `--covar` | `2.0` | `cphm`, `cmm`, `thmm` |
| `--sigma` | `1.0` | `aft_ln` |
| `--shape` | `1.5` | `aft_weibull`, `aft_log_logistic` |
| `--scale` | `2.0` | `aft_weibull`, `aft_log_logistic` |
| `--n-risks` | `2` | competing risks |
| `--baseline-hazards` | — | `competing_risks` — repeat the flag |
| `--shape-params`, `--scale-params` | — | `competing_risks_weibull` — repeat |
| `--cure-fraction`, `--baseline-hazard` | — | `mixture_cure` |
| `--breakpoints`, `--hazard-rates` | — | `piecewise_exponential` — repeat |
| `--process` | `ag` | `recurrent_events` — `ag`, `pwp_tt` or `pwp_gt` |
| `--baseline` | `exponential` | `recurrent_events` — `exponential`, `weibull` or `gompertz` |
| `--rate` | `1.0` | `recurrent_events` — baseline rate, for exponential and Gompertz |
| `--stratum-effects` | — | `recurrent_events` — per-event factors, repeat the flag |
| `--max-events` | `None` | `recurrent_events` — stop a subject after this many events |
| `--followup-time` | `10.0` | `recurrent_events` — administrative end of follow-up |
| `--seed` | `None` | all models |
| `-o` | stdout | output CSV path |

!!! tip "Repeat the flag for list arguments"

    There is no comma syntax. Two coefficients means two `--beta` flags:

    ```bash
    gen_surv dataset aft_weibull --n 200 --beta 0.5 --beta -0.3 \
        --shape 1.5 --scale 2.0 -o aft.csv
    ```

    Same for `--breakpoints`, `--hazard-rates`, `--baseline-hazards`,
    `--shape-params` and `--scale-params`.

### Examples

```bash
# Cox PH, seeded so it is reproducible
gen_surv dataset cphm --n 2000 --beta 0.5 --covariate-range 2.0 \
    --model-cens uniform --cens-par 1.0 --seed 42 -o cphm.csv

# Weibull AFT with two covariates
gen_surv dataset aft_weibull --n 1000 --beta 0.5 --beta -0.3 \
    --shape 1.5 --scale 2.0 --seed 42 -o aft.csv

# Competing risks with three causes
gen_surv dataset competing_risks --n 1000 --n-risks 3 \
    --baseline-hazards 0.3 --baseline-hazards 0.2 --baseline-hazards 0.1 \
    --seed 42 -o cr.csv

# Piecewise exponential, two intervals
gen_surv dataset piecewise_exponential --n 1000 \
    --breakpoints 1.0 --hazard-rates 0.5 --hazard-rates 1.0 \
    --seed 42 -o pw.csv

# Illness-death, counting-process form
gen_surv dataset cmm --n 500 --beta 0.1 --beta 0.2 --beta 0.3 \
    --covariate-range 1.0 \
    --seed 42 -o cmm.csv
```

!!! warning "`--seed` is worth typing every time"

    Without it, the same command gives different data on every run, and there
    is no way to get the first dataset back. See
    [Reproducibility](../getting-started/reproducibility.md).

## `visualize`

Reads a CSV and writes a Kaplan-Meier plot:

```bash
gen_surv visualize data.csv --output km.png
```

| Option | Default | Meaning |
|---|---|---|
| `--time-col` | `time` | column with the observed times |
| `--status-col` | `status` | column with the event indicator |
| `--group-col` | `None` | column to stratify by |
| `--output` | `survival_plot.png` | image path |

```bash
# Stratified by a binary covariate
gen_surv visualize data.csv --group-col X0 --output stratified.png

# For a tdcm file, whose time column is called `stop`
gen_surv visualize tdcm.csv --time-col stop --output tdcm.png
```

Grouping on a continuous covariate produces one curve per distinct value, which
is never what you want. Bin it first in Python — see
[Plotting](plotting.md#stratified-curves).

## Chaining the two

```bash
gen_surv dataset cphm --n 2000 --beta 0.8 --seed 1 -o cphm.csv \
  && gen_surv visualize cphm.csv --output cphm.png
```

## What the CLI cannot do

- **Only CSV out.** For Feather, JSON or RDS, use
  [`export_dataset`](export.md) in Python.
- **No per-covariate distribution control.** `--covariate-dist` does not exist;
  use Python for [`covariate_dist` and `covariate_params`](covariates.md).
- **No summaries.** `describe_survival` and the `gen_surv.summary` functions are
  Python-only — see [Summarising a dataset](summaries.md).

## Related

- [Quickstart](../getting-started/quickstart.md)
- API: [Command line](../api/cli.md)
