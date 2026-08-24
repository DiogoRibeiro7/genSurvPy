# Recurrent events

`model="recurrent_events"` — the same event happening repeatedly to the same
subject: hospital readmissions, asthma attacks, equipment failures, infections.

Unlike every other generator here, a subject does not leave the study when the
event occurs. It stays at risk, and the frame records one interval per event
plus the remainder of follow-up.

## Three processes

The `process` argument picks how the intensity relates to the event history,
matching the three models the data is usually analysed with.

=== "`ag` — Andersen-Gill"

    $$
    \lambda(t \mid X) = h_0(t)\exp(X^\top\beta)
    $$

    The intensity **ignores the event history**: a subject's fourth event is no
    more or less likely than its first, given the same covariates and time.
    Formally a non-homogeneous Poisson process. The clock runs forward from
    entry.

    Use it when events are exchangeable and the covariate effect is what you
    are testing.

=== "`pwp_tt` — PWP, total time"

    $$
    \lambda_k(t \mid X) = h_0(t)\, s_k \exp(X^\top\beta)
    $$

    The intensity of the $k$-th event is scaled by $s_k$, so a second event can
    be more likely than a first. The clock still runs forward from entry.

    Use it when the risk changes with how many events have happened, but the
    baseline is a function of time in the study.

=== "`pwp_gt` — PWP, gap time"

    $$
    \lambda_k(w \mid X) = h_0(w)\, s_k \exp(X^\top\beta),
    \qquad w = t - t_{k-1}
    $$

    As `pwp_tt`, but **the clock resets after every event**, so the baseline is a
    function of time since the previous event.

    Use it when what matters is the waiting time between events rather than
    elapsed study time.

Events are drawn by inversion. With intensity $h_0(t)e^{\eta}s_k$, the
cumulative intensity between consecutive events is $\mathrm{Exponential}(1)$, so
the next event solves

$$
H_0(t) = H_0(t_{k-1}) + \frac{E}{e^{\eta}s_k}
\quad\text{(forward clock)},
\qquad
H_0(w) = \frac{E}{e^{\eta}s_k}
\quad\text{(reset clock)}.
$$

## Baseline hazards

| `baseline` | $h_0(t)$ | Parameters | Shape |
|---|---|---|---|
| `exponential` | $\lambda$ | `rate` | constant |
| `weibull` | $\dfrac{\rho}{\sigma}\left(\dfrac{t}{\sigma}\right)^{\rho-1}$ | `shape`, `scale` | monotone |
| `gompertz` | $a e^{bt}$ | `rate`, `shape` | exponentially rising or falling |

`baseline_params` accepts only the keys belonging to the chosen family; a
misspelt one raises rather than silently leaving the default in place. A
Gompertz `shape` may be **negative**, which gives a declining hazard and a
finite total hazard, so events eventually stop.

## Parameters

```python
gen_recurrent_events(n, process="ag", baseline="exponential",
                     baseline_params=None, betas=None, n_covariates=2,
                     covariate_dist="normal", covariate_params=None,
                     stratum_effects=None, max_events=None,
                     followup_time=10.0, model_cens="uniform",
                     cens_par=20.0, seed=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | — | number of subjects — **not** the number of rows |
| `process` | `"ag"` \| `"pwp_tt"` \| `"pwp_gt"` | `"ag"` | the event process, as above |
| `baseline` | `"exponential"` \| `"weibull"` \| `"gompertz"` | `"exponential"` | baseline hazard family |
| `baseline_params` | `dict` \| `None` | `None` | that family's parameters; defaults filled in |
| `betas` | `list[float]` \| `None` | `None` | coefficients on the log intensity; **drawn at random when omitted** |
| `n_covariates` | `int` | `2` | covariate count when `betas` is omitted |
| `covariate_dist` | `"normal"` \| `"uniform"` \| `"binary"` | `"normal"` | see [Covariates](../guides/covariates.md) |
| `covariate_params` | `dict` \| `None` | `None` | that distribution's parameters |
| `stratum_effects` | `list[float]` \| `None` | `None` | per-event intensity factors, for the PWP processes |
| `max_events` | `int` \| `None` | `None` | stop following a subject after this many events |
| `followup_time` | `float` | `10.0` | administrative end of follow-up, for everyone |
| `model_cens` | `"uniform"` \| `"exponential"` | `"uniform"` | random dropout, on top of `followup_time` |
| `cens_par` | `float` | `20.0` | dropout parameter |
| `seed` | `int` \| `Generator` \| `None` | `None` | reproducibility |

`stratum_effects` applies its **last entry to all later events**, so
`[1.0, 2.0]` reads as "the first event at the baseline rate, every subsequent
one at twice that".

!!! warning "`stratum_effects` with `process="ag"` raises"

    An Andersen-Gill intensity cannot depend on the event number — that is what
    distinguishes it from PWP. Applying the effects anyway would produce PWP
    data under an AG label, and ignoring them would discard an argument you
    clearly meant, so the combination is rejected:

    ```text
    ParameterError: Invalid value for 'stratum_effects': [1.0, 2.0] (type list).
    is not applicable to process='ag', whose intensity cannot depend on the
    event number; use process='pwp_tt' or process='pwp_gt'.
    ```

## Example

```python
from gen_surv import generate

df = generate(model="recurrent_events", n=3,
              baseline_params={"rate": 0.5}, betas=[0.4, -0.2],
              followup_time=5.0, cens_par=1e9, seed=42)
print(df.round(4))
```

```text
 id  start   stop  status  enum      X0      X1
  0 0.0000 1.5050       1     1  0.3047 -1.0400
  0 1.5050 1.6063       1     2  0.3047 -1.0400
  0 1.6063 3.1723       1     3  0.3047 -1.0400
  0 3.1723 5.0000       0     4  0.3047 -1.0400
  1 0.0000 0.6918       1     1  0.7505  0.9406
  1 0.6918 2.8938       1     2  0.7505  0.9406
  1 2.8938 3.1687       1     3  0.7505  0.9406
  1 3.1687 3.3325       1     4  0.7505  0.9406
  1 3.3325 3.8960       1     5  0.7505  0.9406
  1 3.8960 5.0000       0     6  0.7505  0.9406
  2 0.0000 1.3893       1     1 -1.9510 -1.3022
  2 1.3893 5.0000       0     2 -1.9510 -1.3022
```

Three subjects, twelve rows, three and five and one events.

| Column | dtype | Meaning |
|---|---|---|
| `id` | `int64` | subject, from 0 |
| `start`, `stop` | `float64` | the interval over which the subject was at risk of its `enum`-th event |
| `status` | `int64` | `1` if that event occurred at `stop`, `0` if follow-up ended first |
| `enum` | `int64` | which event this interval is about, from 1 |
| `X0`, `X1`, … | `float64` | covariates, constant within a subject |

The intervals **tile each subject's follow-up**: `start` is 0 for the first row
and equal to the previous `stop` thereafter, and the last row of every subject
is censored. Counting subjects means `df["id"].nunique()`, never `len(df)`.

Counting events:

```python
df.groupby("id")["status"].sum()      # events per subject
df["status"].sum()                    # events in total
```

## Check: does a Cox model recover the coefficients?

The frame is already in the `(start, stop]` layout that a time-varying Cox
model expects, so it goes straight into `CoxTimeVaryingFitter`:

```python
from gen_surv import generate
from lifelines import CoxTimeVaryingFitter

df = generate(model="recurrent_events", n=4000,
              baseline_params={"rate": 0.5}, betas=[0.4, -0.2],
              followup_time=5.0, cens_par=1e9, seed=7)

fit = CoxTimeVaryingFitter().fit(df, id_col="id", start_col="start",
                                 stop_col="stop", event_col="status")
print(fit.summary[["coef", "se(coef)"]].round(3))
```

```text
            coef  se(coef)
covariate
enum      -0.004     0.007
X0         0.399     0.011
X1        -0.201     0.010
```

`X0` and `X1` come back at 0.399 and −0.201 against a truth of 0.4 and −0.2.
And because `enum` was left in the frame, it is fitted too — its coefficient of
−0.004 is the Andersen-Gill assumption showing up in the data: **the event
number carries no information**, which is exactly what `process="ag"` promises.
Generate with `pwp_tt` instead and that coefficient stops being zero.

## Check: the counting process itself

With no covariate effect, the expected number of events by time $T$ is the
integrated baseline hazard, $\mathbb{E}[N(T)] = H_0(T)$:

```python
counts = df.groupby("id")["status"].sum()
counts.mean(), counts.var()
```

| Baseline | Parameters | $H_0(5)$ | Observed mean |
|---|---|---|---|
| `exponential` | `rate=0.5` | 2.500 | 2.484 |
| `weibull` | `shape=1.5, scale=2.0` | 3.953 | 3.923 |
| `gompertz` | `rate=0.3, shape=0.2` | 2.577 | 2.562 |

(20,000 subjects, `betas=[0, 0]`, no dropout.)

For the exponential baseline the process is Poisson, so the **variance should
equal the mean** — measured at 2.499 against a mean of 2.484. A generator that
merely got the average right would fail that check.

## Choosing between the clocks

The clock only matters when the baseline is not constant. A constant hazard is
memoryless, so `ag` and `pwp_gt` produce identical data:

```python
common = dict(n=300, baseline_params={"rate": 0.8}, betas=[0.2, -0.1],
              followup_time=6.0, seed=9)

generate(model="recurrent_events", process="ag", **common)
generate(model="recurrent_events", process="pwp_gt", **common)   # the same frame
```

With a rising Weibull hazard they diverge sharply. On a forward clock the
hazard keeps climbing with time in the study; on a reset clock every event puts
the subject back at the start of the curve:

| `shape=2.0, scale=2.0`, `followup_time=6` | Mean events per subject |
|---|---|
| `process="ag"` (forward clock) | 7.18 |
| `process="pwp_gt"` (reset clock) | 2.53 |

## Stratum effects

```python
df = generate(model="recurrent_events", n=20000, process="pwp_gt",
              baseline_params={"rate": 1.0}, betas=[0.0, 0.0],
              stratum_effects=[1.0, 2.0], followup_time=20.0,
              cens_par=1e9, seed=7)

events = df[df["status"] == 1].assign(gap=lambda d: d["stop"] - d["start"])
events.groupby(events["enum"] > 1)["gap"].mean()
```

```text
enum > 1
False    0.982      # first events: mean gap 1 / 1.0
True     0.488      # later events:  mean gap 1 / 2.0
```

Doubling the intensity halves the mean gap, as it should.

## Ending follow-up

Three mechanisms, applied together:

| Mechanism | Argument | Effect |
|---|---|---|
| Administrative | `followup_time` | everyone's follow-up ends here |
| Random dropout | `model_cens`, `cens_par` | subject-specific, on top of the above |
| Event cap | `max_events` | the subject leaves after this many events |

`max_events` ends follow-up **at the capped event**, so a capped subject has no
trailing censored row. That keeps the data honest: the alternative — recording
further at-risk time while suppressing the events in it — would look like a
subject that stopped having events.

## Related

- [Output schemas](../getting-started/schemas.md#recurrent_events) — the counting-process layout
- [Illness-death, intervals (CMM)](cmm.md) — the same layout for transitions between states rather than repeats of one event
- [Competing risks](competing-risks.md) — several kinds of event, each occurring at most once
- API: [`gen_recurrent_events`](../api/generators.md#gen_surv.recurrent.gen_recurrent_events)
