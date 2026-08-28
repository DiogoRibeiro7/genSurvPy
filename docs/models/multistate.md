# The multistate engine

`gen_multistate` walks a subject through an **arbitrary graph of states**. Each
edge carries its own baseline hazard and its own coefficients, so the intensity
of the $i \to j$ transition is

$$
\alpha_{ij}(t \mid X) = h_{0,ij}(t)\exp(X^\top\beta_{ij}).
$$

[CMM](cmm.md) and [THMM](thmm.md) are configurations of it rather than separate
implementations: the illness-death graph, with Weibull or exponential edges.

## Describing a graph

A `Transition` is one edge: where from, where to, the baseline hazard, and the
coefficients acting on it.

```python
from gen_surv import ExponentialBaseline, Transition, WeibullBaseline, gen_multistate

transitions = [
    Transition(1, 2, WeibullBaseline(shape=1.2, scale=3.0), [0.4]),   # onset
    Transition(1, 3, ExponentialBaseline(rate=0.2), [0.1]),           # death, healthy
    Transition(2, 3, WeibullBaseline(shape=0.8, scale=1.5), [0.6]),   # death, ill
]

df = gen_multistate(n=500, transitions=transitions, clock="reset",
                    cens_par=20.0, max_time=15.0, seed=1)
```

A state with **no outgoing transition is absorbing**: follow-up ends when a
subject reaches it. Above, state 3.

Cycles are allowed. Recovery is a transition like any other:

```python
transitions = [
    Transition(1, 2, ExponentialBaseline(0.8), [0.0]),   # fall ill
    Transition(2, 1, ExponentialBaseline(0.6), [0.0]),   # recover
    Transition(2, 3, ExponentialBaseline(0.2), [0.0]),   # die while ill
]
```

Subjects then move between 1 and 2 repeatedly until they reach 3 or follow-up
ends.

## The clock

The one modelling choice that has no counterpart in a single-event model.

| `clock` | The hazard is a function of | Process |
|---|---|---|
| `"forward"` | time since entry to the **study** | Markov |
| `"reset"` | time since entry to the **current state** | semi-Markov |

With an exponential baseline the two coincide, because a constant hazard is
memoryless — the same seed even gives the identical frame. With a rising
Weibull they diverge sharply. Measured over 8,000 subjects with
`shape=2.0, scale=3.0` on both edges:

| Clock | Mean `2 → 3` sojourn |
|---|---|
| `forward` | 1.33 |
| `reset` | 2.66 |

On a forward clock a subject entering state 2 finds the hazard already part-way
up its curve; on a reset clock it starts at the bottom again.

## Parameters

```python
gen_multistate(n, transitions, clock="forward", initial_state=1,
               covariate_dist="normal", covariate_params=None,
               model_cens="uniform", cens_par=5.0, max_time=None,
               layout="intervals", seed=None)
```

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `n` | `int` | — | number of subjects — **not** the number of rows |
| `transitions` | `Sequence[Transition]` | — | the graph |
| `clock` | `"forward"` \| `"reset"` | `"forward"` | as above |
| `initial_state` | `int` | `1` | the state every subject starts in at time 0 |
| `covariate_dist` | `"normal"` \| `"uniform"` \| `"binary"` | `"normal"` | see [Covariates](../guides/covariates.md) |
| `covariate_params` | `dict` \| `None` | `None` | that distribution's parameters |
| `model_cens` | `"uniform"` \| `"exponential"` | `"uniform"` | random dropout |
| `cens_par` | `float` | `5.0` | dropout parameter |
| `max_time` | `float` \| `None` | `None` | administrative end of follow-up |
| `layout` | `"intervals"` \| `"panel"` | `"intervals"` | which canonical schema to return |
| `seed` | `int` \| `Generator` \| `None` | `None` | reproducibility |

Every transition must carry the **same number of coefficients**, one per
covariate, and an origin-destination pair may appear only once.

## Both layouts

=== "`intervals`"

    One row per transition a subject was at risk of, over the interval it was
    at risk. `status` marks the one that occurred.

    ```text
     id  start   stop  from_state  to_state  status      X0
      0 0.0000 4.5060           1         2       0  0.3047
      0 0.0000 4.5060           1         3       1  0.3047
      1 0.0000 1.2896           1         2       1 -1.0400
      1 0.0000 1.2896           1         3       0 -1.0400
      1 1.2896 2.6012           2         3       1 -1.0400
    ```

=== "`panel`"

    One row per observation of the subject's state: an entry at time 0, one per
    transition, and one in the state occupied when follow-up ends.

    ```text
     id   time  state      X0
      0 0.0000      1  0.3047
      0 0.2643      2  0.3047
      0 0.4475      3  0.3047
      1 0.0000      1 -1.0400
      1 1.9345      3 -1.0400
    ```

These are the two contracts described in
[Output schemas](../getting-started/schemas.md#the-two-canonical-layouts).

## Check: are the intensities the ones you set?

Events divided by time at risk, per edge — which the interval layout gives
directly:

```python
from gen_surv import ExponentialBaseline, Transition, gen_multistate

rates = {(1, 2): 0.3, (1, 3): 0.2, (2, 3): 0.5}
transitions = [Transition(o, d, ExponentialBaseline(r), [0.0])
               for (o, d), r in rates.items()]

df = gen_multistate(n=40000, transitions=transitions, cens_par=1e9,
                    max_time=60.0, seed=7)

for (origin, destination), declared in rates.items():
    rows = df[(df["from_state"] == origin) & (df["to_state"] == destination)]
    exposure = float((rows["stop"] - rows["start"]).sum())
    print(f"{origin}->{destination}  declared={declared}  "
          f"mle={int(rows['status'].sum()) / exposure:.3f}")
```

```text
1->2  declared=0.3  mle=0.299
1->3  declared=0.2  mle=0.198
2->3  declared=0.5  mle=0.501
```

Two further properties worth knowing, both of which follow from competing
exponentials: the sojourn in state 1 is exponential with the **summed**
intensity, and the probability of leaving to state 2 rather than 3 is
$\lambda_{12} / (\lambda_{12} + \lambda_{13})$ — measured at 0.5953 against a
theoretical 0.6.

## Not reachable through `generate()`

Every other model is addressed by a string, but a graph is a list of objects
rather than a set of scalars, so there is no `generate(model="multistate", ...)`
and no command-line equivalent. Import `gen_multistate` directly.

## Related

- [Illness-death, intervals (CMM)](cmm.md) — the engine with Weibull edges and a reset clock
- [Illness-death, panel (THMM)](thmm.md) — exponential edges, panel layout
- [Baseline hazards](../guides/baselines.md) — what an edge can carry
- API: [`gen_multistate`](../api/generators.md#gen_surv.multistate)
