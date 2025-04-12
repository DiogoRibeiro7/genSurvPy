# 📘 Mathematical Foundations of `gen_surv`

This page presents the mathematical formulation behind the survival models implemented in the `gen_surv` package.

---

## 1. Cox Proportional Hazards Model (CPHM)

The hazard function conditioned on covariates is:

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

A generalization of CPHM where covariates change over time:

$$
h(t \mid Z(t)) = h_0(t) \\exp(Z(t) \\beta)
$$

In this package, piecewise covariate values are simulated with dependence across segments using correlated normal draws.

---

## 3. Continuous-Time Multi-State Markov Model (CMM)

Markov model with generator matrix \( Q \). The transition probability matrix is given by:

$$
P(t) = \\exp(Qt)
$$

Where:

- \( Q \) is the rate matrix
- \( P(t)_{ij} \) gives the probability of being in state j at time t given starting in state i

---

## 4. Time-Homogeneous Hidden Markov Model (THMM)

This model simulates observed states with unobserved latent state transitions.

Let:

- \( S_t \) be the latent state at time t
- \( Y_t \) be the observed variable conditional on \( S_t \)

The transition structure is governed by a homogeneous Markov chain with transition matrix \( P \), and emissions are Gaussian:

$$
Y_t \mid S_t = k \\sim \\mathcal{N}(\\mu_k, \\sigma_k^2)
$$

---

## Notes

All models support censoring:

- **Uniform:** \( C_i \\sim U(0, \\text{cens\\_par}) \)
- **Exponential:** \( C_i \\sim \\text{Exp}(\\text{cens\\_par}) \)
