# Mathematical foundations

The formulation behind each generator. Every model page repeats the part
relevant to it; this page puts them side by side and fixes notation.

Throughout, $T$ is the event time, $C$ the censoring time, $X$ a covariate
vector, and the observed pair is $(\min(T, C), \mathbb{1}\{T \le C\})$.

Three equivalent ways to describe a survival distribution:

$$
S(t) = \Pr(T > t), \qquad
h(t) = \lim_{\Delta \to 0}\frac{\Pr(t \le T < t + \Delta \mid T \ge t)}{\Delta},
\qquad
S(t) = \exp\!\left(-\int_0^t h(u)\,du\right).
$$

Sampling works through the last of these: draw $U \sim \mathrm{Uniform}(0,1)$
and solve $S(T) = U$. Every generator here is some version of that inversion.

## 1. Cox proportional hazards

The hazard splits into a baseline and a covariate multiplier:

$$
h(t \mid X) = h_0(t)\exp(X^\top\beta).
$$

The ratio of hazards for two subjects is $\exp((X_1 - X_2)^\top\beta)$ — constant
in $t$, which is what "proportional" means. `cphm` uses a constant baseline
$h_0(t) = 1$, so

$$
T \mid X \sim \mathrm{Exponential}\big(\exp(X^\top\beta)\big),
$$

drawn by inversion as $T = -\log(U)/\exp(X^\top\beta)$.

With a Weibull baseline $h_0(t) = \lambda\rho t^{\rho-1}$ the cumulative hazard
is $\Lambda_0(t) = \lambda t^{\rho}$ and

$$
S(t \mid X) = \exp\!\big(-\lambda t^{\rho}\exp(X^\top\beta)\big),
$$

which is what the [Weibull AFT generator](../models/aft.md) produces in its
PH parameterisation.

→ [`cphm`](../models/cphm.md)

## 2. Accelerated failure time

Instead of scaling the hazard, scale time:

$$
\log T = X^\top\beta + \sigma\varepsilon,
$$

so $S(t \mid X) = S_0\!\left(t e^{-X^\top\beta}\right)$: a covariate stretches or
compresses the whole survival curve. The distribution of $\varepsilon$ picks the
family.

**Log-normal.** $\varepsilon \sim \mathcal{N}(0,1)$, giving

$$
S(t \mid X) = 1 - \Phi\!\left(\frac{\log t - X^\top\beta}{\sigma}\right),
$$

with $\Phi$ the standard normal CDF. The hazard rises then falls.

**Weibull.** The only family that is both AFT and PH. `gen_surv` draws it as

$$
T = \texttt{scale}\cdot\big(-\log U \cdot e^{-X^\top\beta}\big)^{1/\texttt{shape}},
$$

which is the **PH** parameterisation — $\beta$ is a log hazard ratio, and its
effect on $\log T$ is $-\beta/\texttt{shape}$.

**Log-logistic.** $S(t) = \big(1 + (t/\texttt{scale})^{\texttt{shape}}\big)^{-1}$,
giving a unimodal hazard: rising to a peak, then decaying. Not a PH family.

→ [AFT models](../models/aft.md)

## 3. Piecewise exponential

Partition follow-up at $0 < \tau_1 < \dots < \tau_k$ and hold the hazard
constant on each piece:

$$
h_0(t) = \lambda_j, \quad t \in [\tau_j, \tau_{j+1}).
$$

The cumulative hazard is piecewise linear,

$$
\Lambda_0(t) = \sum_{j} \lambda_j \big(\min(t, \tau_{j+1}) - \tau_j\big)_+,
$$

and inversion walks the intervals, consuming exponential "budget" until it runs
out inside one of them. With enough pieces this approximates any hazard shape.

→ [`piecewise_exponential`](../models/piecewise-exponential.md)

## 4. Competing risks

With $K$ causes, each has a **cause-specific hazard**

$$
h_k(t \mid X) = \lim_{\Delta \to 0}
\frac{\Pr(t \le T < t+\Delta,\; \delta = k \mid T \ge t)}{\Delta}
= h_{0k}(t)\exp(X^\top\beta_k).
$$

The all-cause hazard is $\sum_k h_k$, and the observed pair is
$(T, \delta) = (\min_k T_k, \arg\min_k T_k)$.

The quantity usually reported is the cumulative incidence

$$
F_k(t) = \int_0^t h_k(u) S(u)\, du,
$$

which depends on **all** the hazards through $S$ — this is why a Fine-Gray
subdistribution coefficient does not equal the cause-specific $\beta_k$ that
generated the data.

→ [Competing risks](../models/competing-risks.md)

## 5. Mixture cure

A latent indicator $Y$ marks the uncured, with $\Pr(Y = 1 \mid X)$ logistic in
$X$. The population survival function mixes a point mass at infinity with a
proper distribution:

$$
S_{\text{pop}}(t \mid X) = \pi(X) + \big(1 - \pi(X)\big)S_u(t \mid X),
$$

where $\pi(X)$ is the cure probability and $S_u$ the survival function of the
uncured. As $t \to \infty$, $S_{\text{pop}} \to \pi(X)$ — the plateau. `gen_surv`
takes $S_u$ exponential with hazard $\lambda\exp(X^\top\beta_{\text{surv}})$.

→ [Mixture cure](../models/mixture-cure.md)

## 6. Multi-state models

States $\{1, 2, 3\}$ = healthy, ill, dead, with transition intensities

$$
\alpha_{ij}(t \mid X) = \lim_{\Delta \to 0}
\frac{\Pr\big(Z(t + \Delta) = j \mid Z(t) = i, X\big)}{\Delta}.
$$

For a **time-homogeneous** chain the intensities are constant, the generator
matrix $Q$ has $Q_{ij} = \alpha_{ij}$ and $Q_{ii} = -\sum_{j \ne i}\alpha_{ij}$,
and transition probabilities follow from the matrix exponential:

$$
P(t) = \exp(Qt).
$$

Sojourns are exponential and memoryless. That is [`thmm`](../models/thmm.md),
with $\alpha_{ij} = \lambda_{ij}\exp(\beta_{ij}X)$.

[`cmm`](../models/cmm.md) instead draws Weibull sojourns,

$$
T_{ij} = \left(\frac{-\log(1-U)}{\lambda_{ij}\exp(\beta_{ij}X)}\right)^{1/\rho_{ij}},
$$

on a clock that **resets** when a subject enters state 2. The intensity of
dying then depends on time since illness rather than time since entry, making
the process semi-Markov rather than Markov.

## 7. Time-dependent covariates

When a covariate changes during follow-up the hazard is

$$
h(t \mid Z(t)) = h_0(t)\exp(Z(t)^\top\beta),
$$

and the partial likelihood needs $Z(t)$ evaluated at each event time — hence
the `(start, stop]` layout that survival software expects. `gen_surv` simulates
a single switch from 0 to 1 at a crossover time correlated with the baseline
covariate.

→ [`tdcm`](../models/tdcm.md)

## Censoring

All generators apply independent right-censoring: $C \perp T \mid X$, with

$$
C \sim \mathrm{Uniform}(0, \texttt{cens\_par})
\quad\text{or}\quad
C \sim \mathrm{Exponential}(\text{mean} = \texttt{cens\_par}).
$$

Independence is what makes the standard estimators unbiased here. To break it
deliberately, see [Censoring](../guides/censoring.md#dependent-censoring).

Some models add administrative censoring at `max_time`, which is deterministic
and also independent of $T$.

## Further reading

See the [bibliography](bibliography.md) for the sources behind each of these.
