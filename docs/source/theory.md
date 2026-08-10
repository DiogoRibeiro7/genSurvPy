---
orphan: true
---

# 📘 Mathematical Foundations of `gen_surv`

This page presents the mathematical formulation behind the survival models implemented in the `gen_surv` package.

---

## 1. Cox Proportional Hazards Model (CPHM)

This semi-parametric approach models the hazard as a baseline component
multiplied by an exponential term involving the covariates. The hazard
function conditioned on covariates is:

$$
h(t \mid X) = h_0(t) \exp(X \\beta)
$$

Where:

- \( h_0(t) \) is the baseline hazard
- \( X \\beta \) is the linear predictor

### Weibull baseline hazard:

$$
h_0(t) = \\lambda \\rho t^{\\rho - 1}
$$

The cumulative hazard is:

$$
\\Lambda_0(t) = \\lambda t^{\\rho}
$$

And the survival function becomes:

$$
S(t \mid X) = \\exp\\left(-\\Lambda_0(t) \\exp(X \\beta)\\right)
$$

---

## 2. Time-Dependent Covariate Model (TDCM)

This extension of the Cox model allows covariate values to vary during
follow-up, accommodating exposures or treatments that change over time:

$$
h(t \mid Z(t)) = h_0(t) \\exp(Z(t) \\beta)
$$

In this package, piecewise covariate values are simulated with dependence across segments using correlated normal draws.

---

## 3. Continuous-Time Multi-State Markov Model (CMM)

This framework captures transitions between a finite set of states where
waiting times are exponentially distributed. With generator matrix \( Q \), the transition probability matrix is given by:

$$
P(t) = \\exp(Qt)
$$

Where:

- \( Q \) is the rate matrix
- \( P(t)_{ij} \) gives the probability of being in state j at time t given starting in state i

---

## 4. Time-Homogeneous Markov Model (THMM)

This model describes a subject moving through three observed states,
1 (healthy), 2 (illness) and 3 (death), where every transition intensity is
constant in time. That constancy is what "time-homogeneous" refers to; the
states themselves are observed, so this is not a hidden Markov model.

For a covariate \( X \), the intensity of the transition from state \( i \) to
state \( j \) is

$$
\alpha_{ij}(t \mid X) = \lambda_{ij} \exp(\beta_{ij} X)
$$

with \( \lambda_{ij} > 0 \) constant in \( t \). The three rates and three
coefficients correspond to the \( 1 \to 2 \), \( 1 \to 3 \) and \( 2 \to 3 \)
transitions.

Because the intensities do not depend on \( t \), each candidate sojourn time is
exponential:

$$
T_{ij} \mid X \sim \mathrm{Exponential}\!\left(\lambda_{ij} \exp(\beta_{ij} X)\right)
$$

A subject leaves state 1 at \( \min(T_{12}, T_{13}) \), and the destination is
whichever of the two came first. If censoring occurs before that, the subject is
recorded in state 1.

```{note}
The generator currently reports only the first transition, so it emits one row
per subject and the `2 -> 3` intensity does not appear in the returned data.
Emitting the full trajectory is planned; see the roadmap.
```

---


## 5. Accelerated Failure Time (AFT) Models

These fully parametric models relate covariates to the logarithm of the
survival time. They assume the effect of a covariate speeds up or slows down the
event time directly, rather than acting on the hazard.

### Log-Normal AFT

The model assumes:

$$
\log(T_i) = X_i \beta + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Thus:

$$
T_i \sim \text{Log-Normal}(X_i \beta, \sigma^2)
$$

The survival function is:

$$
S(t \mid X) = 1 - \Phi\left(\frac{\log(t) - X \beta}{\sigma}\right)
$$

Where:

- \( \Phi \) is the standard normal cumulative distribution function (CDF)

This model is especially useful when the proportional hazards assumption is not valid and provides interpretable effects in the time domain.

## Notes

All models support censoring:

- **Uniform:** \( C_i \sim U(0, \text{cens\_par}) \)
- **Exponential:** \( C_i \sim \text{Exp}(\text{cens\_par}) \)

## 6. Competing Risks Models

These models handle scenarios where several distinct failure types can occur.
Each cause has its own hazard function, and the observed status indicates which
event occurred (1, 2, ...). The package includes constant-hazard and
Weibull-hazard versions.

## 7. Mixture Cure Models

These models posit that a subset of the population is cured and will never
experience the event of interest. The generator mixes a logistic cure component
with an exponential hazard for the uncured, returning a ``cured`` indicator
column alongside the usual time and status.

## 8. Piecewise Exponential Model

Here the baseline hazard is assumed constant within each of several
user-specified intervals. This allows flexible hazard shapes over time while
remaining easy to simulate.
