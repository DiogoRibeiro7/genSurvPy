# Baseline hazards

Every generator that draws a continuous time does the same thing: draw
$E \sim \mathrm{Exponential}(1)$ and solve

$$
H_0(t) = \frac{E}{\exp(X^\top\beta)}
$$

for $t$. The models differ only in which $H_0$ they use.

`gen_surv.baseline` makes that the explicit contract, so a simulator written
against it accepts **any** hazard shape rather than needing a separate entry
point per family.

## The families

| Class | $H_0(t)$ | Parameters | Hazard shape |
|---|---|---|---|
| `ExponentialBaseline` | $\lambda t$ | `rate` | constant |
| `WeibullBaseline` | $(t/\sigma)^{\rho}$ | `shape`, `scale` | monotone |
| `GompertzBaseline` | $\frac{a}{b}\left(e^{bt} - 1\right)$ | `rate`, `shape` | exponentially rising or falling |
| `LogLogisticBaseline` | $\log\!\left(1 + (t/\sigma)^{\rho}\right)$ | `shape`, `scale` | unimodal |
| `PiecewiseConstantBaseline` | piecewise linear | `breakpoints`, `hazard_rates` | constant within intervals |

```python
from gen_surv import WeibullBaseline

baseline = WeibullBaseline(shape=2.0, scale=1.5)

baseline.hazard(3.0)                       # h0(3)
baseline.cumulative_hazard(3.0)            # H0(3)
baseline.inverse_cumulative_hazard(4.0)    # t such that H0(t) = 4
```

All three methods take a scalar or a NumPy array. The classes are frozen
dataclasses that validate on construction, so an invalid shape fails where you
wrote it rather than inside a sampler:

```python
WeibullBaseline(shape=0.0, scale=1.0)
```

```text
PositiveValueError: Argument 'shape' must be greater than 0; got 0.0 of type
float. Try a positive number such as 1.0.
```

## Using one

Anywhere a generator takes a `baseline`, you can pass a name with parameters or
an object:

=== "By name"

    ```python
    from gen_surv import generate

    df = generate(model="recurrent_events", n=500,
                  baseline="weibull",
                  baseline_params={"shape": 1.3, "scale": 2.0},
                  betas=[0.4, -0.2], followup_time=5.0, seed=1)
    ```

=== "As an object"

    ```python
    from gen_surv import generate, WeibullBaseline

    df = generate(model="recurrent_events", n=500,
                  baseline=WeibullBaseline(shape=1.3, scale=2.0),
                  betas=[0.4, -0.2], followup_time=5.0, seed=1)
    ```

The two produce the identical frame — the name is a shortcut for constructing
the object. Passing both a name's `baseline_params` and an object raises, since
the parameters could not be applied without rebuilding it.

The object form is how you reach families a generator does not name.
[`gen_recurrent_events`](../models/recurrent-events.md) knows three names, but
takes any of the five:

```python
from gen_surv import generate, LogLogisticBaseline

baseline = LogLogisticBaseline(shape=2.0, scale=1.5)
df = generate(model="recurrent_events", n=2000, baseline=baseline,
              betas=[0.0, 0.0], followup_time=6.0, cens_par=1e9, seed=3)

df.groupby("id")["status"].sum().mean()   # 2.896
baseline.cumulative_hazard(6.0)           # 2.833
```

With no covariate effect the mean event count is $H_0(T)$, which is the check
that the baseline you passed is really driving the sampler.

## Writing your own

Anything with `hazard`, `cumulative_hazard` and `inverse_cumulative_hazard` is
a baseline. The protocol is runtime-checkable, so `isinstance` works:

```python
from dataclasses import dataclass
import numpy as np
from gen_surv.baseline import BaselineHazard


@dataclass(frozen=True)
class LinearHazard:
    """h0(t) = a * t, so H0(t) = a * t^2 / 2."""

    slope: float

    def hazard(self, t):
        return self.slope * np.asarray(t, dtype=float)

    def cumulative_hazard(self, t):
        return self.slope * np.asarray(t, dtype=float) ** 2 / 2.0

    def inverse_cumulative_hazard(self, value):
        return np.sqrt(2.0 * np.asarray(value, dtype=float) / self.slope)


isinstance(LinearHazard(0.5), BaselineHazard)     # True
```

The one rule that matters: `cumulative_hazard` and
`inverse_cumulative_hazard` must be mutual inverses. If they are not, the times
drawn will be wrong in a way that nothing about the shape of the returned frame
would reveal. Test the round trip:

```python
baseline = LinearHazard(0.5)
t = np.array([0.1, 1.0, 4.0])
np.testing.assert_allclose(
    baseline.inverse_cumulative_hazard(baseline.cumulative_hazard(t)), t
)
```

## When the hazard runs out

A declining Gompertz hazard has a **finite total**: $\lim_{t\to\infty} H_0(t) =
a/|b|$. Past that, no draw of $E$ can ever be consumed, so the event never
happens.

```python
from gen_surv import GompertzBaseline

baseline = GompertzBaseline(rate=0.4, shape=-0.3)
baseline.total_hazard                                  # 1.333…
baseline.inverse_cumulative_hazard(2.0)                # inf
```

`inf` is the correct answer, not an error, and generators treat it as "no
further event" rather than raising. It is what lets a Gompertz baseline express
a subpopulation that simply stops failing.

## Related

- [Recurrent events](../models/recurrent-events.md) — the first generator built on the protocol
- [Piecewise exponential](../models/piecewise-exponential.md) — the same piecewise shape as a standalone model
- API: [Baseline hazards](../api/baselines.md)
