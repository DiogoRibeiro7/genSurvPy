# Output schemas

Every generator returns a `pandas.DataFrame`, but **the frames are not
interchangeable**. Column names, dtypes, whether there is an `id` at all, and
how many rows a subject contributes all vary by model family. This page is the
reference for that.

## The two canonical layouts

Two shapes are contracts rather than conventions. New generators adopt one of
them; the [regression suite](https://github.com/DiogoRibeiro7/genSurvPy/blob/develop/tests/test_generate_regression.py)
pins the exact column list of every model, so a change to any of them fails a
test rather than reaching a release.

**One row per subject.** `time`, `status`, then covariates. The subject leaves
the study when the event occurs. Used by `cphm`, the three AFT models,
`piecewise_exponential`, both competing-risks models and `mixture_cure`.

**Counting-process intervals.** `id`, `start`, `stop`, `status`, then whatever
identifies the interval, then covariates. One row per interval a subject was at
risk over, with `status` marking whether the event closing it occurred. Used by
`cmm` (identified by `from_state` and `to_state`), `recurrent_events`
(identified by `enum`) and `tdcm`. Intervals within a subject are contiguous.

`thmm` is the documented exception: a panel of observed states, which is how
the R package returns it and how multi-state estimators expect it. See
[the note below](#why-cmm-and-thmm-differ).

## At a glance

| `model=` | Rows per subject | Columns |
|---|---|---|
| `cphm` | 1 | `time`, `status`, `X0` |
| `aft_ln`, `aft_weibull`, `aft_log_logistic` | 1 | `id`, `time`, `status`, `X0`, `X1` |
| `piecewise_exponential` | 1 | `id`, `time`, `status`, `X0`, `X1`, … |
| `competing_risks`, `competing_risks_weibull` | 1 | `id`, `time`, `status`, `X0`, `X1`, … |
| `mixture_cure` | 1 | `id`, `time`, `status`, `cured`, `X0`, `X1`, … |
| `cmm` | 2 or 3 | `id`, `start`, `stop`, `from_state`, `to_state`, `status`, `X0` |
| `thmm` | 2 or 3 | `id`, `time`, `state`, `X0` |
| `tdcm` | 1 | `id`, `start`, `stop`, `status`, `covariate`, `tdcov` |
| `recurrent_events` | 1 per at-risk interval | `id`, `start`, `stop`, `status`, `enum`, `X0`, `X1`, … |

!!! danger "Four traps"

    1. **`cphm` has no `id` column.** Every other model has one.
    2. **`status` is `float64` for `cphm` and `tdcm`, `int64` everywhere
       else.** A `df["status"] == 1` comparison works either way, but
       `is` / dtype-sensitive code does not.
    3. **`id` starts at 0 for `cphm`-family and `cmm`, at 1 for `thmm` and
       `tdcm`.**
    4. **`status` is not always a 0/1 event indicator.** For competing risks
       it is the *cause* (`0` censored, `1`, `2`, …). For `cmm` it marks
       which candidate transition actually happened.

## One row per subject

### `cphm`

```text
       time  status        X0
0  0.438878     0.0  1.547912
1  0.094177     0.0  1.394736
2  0.037041     1.0  1.522279
```

| Column | dtype | Meaning |
|---|---|---|
| `time` | `float64` | `min(event time, censoring time)` |
| `status` | `float64` | `1.0` event observed, `0.0` censored |
| `X0` | `float64` | covariate, `Uniform(0, covariate_range)` |

### AFT models

`aft_ln`, `aft_weibull` and `aft_log_logistic` share a schema. The number of
covariates follows the length of `beta`.

```text
 id     time  status        X0        X1
  0 1.699586       1  0.304717 -1.039984
  1 1.238956       0  0.750451  0.940565
  2 0.889270       1 -1.951035 -1.302180
```

| Column | dtype | Meaning |
|---|---|---|
| `id` | `int64` | subject index, `0 … n-1` |
| `time` | `float64` | observed time |
| `status` | `int64` | `1` event, `0` censored |
| `X0`, `X1`, … | `float64` | covariates, standard normal, one per entry in `beta` |

### `piecewise_exponential`

Same shape as AFT. Covariate count comes from `n_covariates` (default 2) or the
length of `betas`.

```text
 id     time  status        X0        X1
  0 3.790439       0  0.750451  0.940565
  1 1.772630       0 -1.951035 -1.302180
  2 0.980989       1  0.127840 -0.316243
```

### Competing risks

`status` carries the **cause**, not a binary indicator.

```text
 id     time  status        X0        X1
  0 0.125269       1 -1.951035 -1.302180
  3 0.832001       2  0.879398  0.777792
```

| Value of `status` | Meaning |
|---|---|
| `0` | censored — no cause occurred before censoring or `max_time` |
| `1` | cause 1 occurred |
| `2` | cause 2 occurred |
| `k` | cause `k` occurred, up to `n_risks` |

To analyse one cause, build your own indicator:

```python
from gen_surv import generate

df = generate(model="competing_risks", n=100, n_risks=2, seed=1)

cause1 = df.assign(event=(df["status"] == 1).astype(int))
```

### `mixture_cure`

Adds a `cured` column that no other model has.

```text
 id     time  status  cured        X0        X1
  0 0.973194       0      0 -1.951035 -1.302180
  2 0.219019       0      1 -0.016801 -0.853044
```

| Column | dtype | Meaning |
|---|---|---|
| `cured` | `int64` | `1` if the subject belongs to the cured fraction and will never have the event |
| `status` | `int64` | `1` event observed, `0` censored — a cured subject is always `0` |

`cured` is ground truth you would never have in a real study. It is there so you
can check whether a cure-model estimator identifies the right people.

## Several rows per subject

### `cmm` — counting-process intervals

```text
 id  start     stop  from_state  to_state  status       X0
  0    0.0 1.657059           1         2       0 0.773956
  0    0.0 1.657059           1         3       1 0.773956
  1    0.0 0.704256           1         2       0 0.438878
  1    0.0 0.704256           1         3       1 0.438878
```

One row per **transition the subject was at risk of**, not per transition that
happened. While in state 1 a subject can go to either state 2 or state 3, so it
contributes a row for each over the same `[start, stop)` interval, and `status`
marks which one actually occurred. A subject that reaches state 2 contributes a
further `2 -> 3` row.

| Column | dtype | Meaning |
|---|---|---|
| `id` | `int64` | subject, from 0 |
| `start`, `stop` | `float64` | the at-risk interval for this transition |
| `from_state`, `to_state` | `int64` | the transition this row is about: `1→2`, `1→3` or `2→3` |
| `status` | `int64` | `1` if this transition is the one that occurred at `stop` |
| `X0` | `float64` | covariate, constant within a subject |

So a subject contributes **two rows** if it never leaves state 1, and **three**
if it reaches state 2. Never assume one row per subject:

```python
from gen_surv import generate

df = generate(model="cmm", n=100, model_cens="uniform", cens_par=2.0,
              beta=[0.1, 0.2, 0.3], covariate_range=1.0,
              rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0], seed=1)

n_subjects = df["id"].nunique()      # not len(df)
```

### `thmm` — observed state panel

```text
 id     time  state       X0
  1 0.000000      1 0.773956
  1 1.104706      3 0.773956
  2 0.000000      1 0.438878
  2 0.469504      3 0.438878
```

One row per **observation of the subject's state**: an entry observation in
state 1 at time 0, then one row per transition, or one row in whichever state
the subject occupies at censoring.

| Column | dtype | Meaning |
|---|---|---|
| `id` | `int64` | subject, from 1 |
| `time` | `float64` | the time this state was observed |
| `state` | `int64` | `1` healthy, `2` ill, `3` dead |
| `X0` | `float64` | covariate, constant within a subject |

There is **no `status` column**. Whether a subject was censored is read off its
last state: an absorbing final state of 3 is a death, anything else is
censoring.

### Why CMM and THMM differ

!!! note ""

    The two layouts are deliberate, matching the R package's split between
    `genCMM` (transition intervals) and `genTHMM` (states observed at times).
    Both describe the same illness-death process. Pick the layout your
    estimator expects — see [CMM](../models/cmm.md) and
    [THMM](../models/thmm.md).

### `tdcm`

```text
 id  start     stop  status  covariate  tdcov
1.0    0.0 0.806478     1.0   0.494017    0.0
2.0    0.0 0.978989     1.0   0.748251    1.0
```

| Column | dtype | Meaning |
|---|---|---|
| `id` | `float64` | subject, from 1 — note the dtype |
| `start`, `stop` | `float64` | the interval this row covers |
| `status` | `float64` | `1.0` event at `stop`, `0.0` censored |
| `covariate` | `float64` | the baseline covariate |
| `tdcov` | `float64` | the time-dependent covariate's value over the interval |

This is the only model that names its covariate `covariate` rather than `X0`.

### `recurrent_events`

```text
 id  start   stop  status  enum      X0      X1
  0 0.0000 1.5050       1     1  0.3047 -1.0400
  0 1.5050 1.6063       1     2  0.3047 -1.0400
  0 1.6063 3.1723       1     3  0.3047 -1.0400
  0 3.1723 5.0000       0     4  0.3047 -1.0400
  2 0.0000 1.3893       1     1 -1.9510 -1.3022
  2 1.3893 5.0000       0     2 -1.9510 -1.3022
```

One row per **at-risk interval**: the subject stays in the study after each
event, so a subject with three events contributes four rows — three ending in an
event, then the remainder of follow-up.

| Column | dtype | Meaning |
|---|---|---|
| `id` | `int64` | subject, from 0 |
| `start`, `stop` | `float64` | the interval over which the subject was at risk of its `enum`-th event |
| `status` | `int64` | `1` if that event occurred at `stop`, `0` if follow-up ended first |
| `enum` | `int64` | which event this interval is about, from 1 |
| `X0`, `X1`, … | `float64` | covariates, constant within a subject |

The intervals tile each subject's follow-up: `start` is 0 on the first row and
equal to the previous `stop` afterwards, and the final row is censored — unless
`max_events` ended follow-up at a capped event. Row counts are unbounded, so
this is the model most likely to break code that assumes one row per subject.

## Writing code that survives a model change

```python
def event_count(df):
    """Works for cphm, aft_*, piecewise, mixture_cure — not competing risks."""
    return int((df["status"] == 1).sum())


def subject_count(df):
    """Works for every model."""
    return df["id"].nunique() if "id" in df else len(df)
```

If you need one function across all twelve, branch on the columns present
rather than on the model name — `cmm` is the frame with `from_state`, `thmm` the
one with `state` and no `status`, `mixture_cure` the one with `cured`, and
`recurrent_events` the one with `enum`.
