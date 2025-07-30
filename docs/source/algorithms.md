---
orphan: true
---

# Algorithm Overview

This page provides a short description of each model implemented in **gen_surv**.  For mathematical details see {doc}`theory`.

## Cox Proportional Hazards Model (CPHM)
The hazard at time $t$ is proportional to a baseline hazard multiplied by the exponential of covariate effects.
It is widely used for modelling relative risks under the proportional hazards assumption.
See {ref}`Cox1972` in the {doc}`bibliography` for the seminal paper.

## Accelerated Failure Time Models (AFT)
These parametric models directly relate covariates to survival time.
gen_surv includes log-normal, log-logistic and Weibull variants allowing different baseline distributions.
They are convenient when the effect of covariates accelerates or decelerates event times.

## Continuous-Time Multi-State Markov Model (CMM)
Transitions between states are governed by a generator matrix.
This model is suited for illness-death and other multi-state processes where state occupancy changes continuously over time.
The mathematical formulation follows the counting-process approach of Andersen et al. {ref}`Andersen1993`.

## Time-Dependent Covariate Model (TDCM)
Extends the Cox model to covariates that vary during follow-up.
Covariates are simulated in a piecewise fashion with optional correlation across segments.

## Time-Homogeneous Hidden Markov Model (THMM)
Handles processes with unobserved states that emit observable values.
The latent transitions follow a homogeneous Markov chain while emissions are Gaussian.
For background on these models see Zucchini et al. {ref}`Zucchini2017`.

## Competing Risks
Allows multiple failure types with cause-specific hazards.
gen_surv supports constant and Weibull hazards for each cause.
The subdistribution approach of Fine and Gray {ref}`FineGray1999` is commonly used for analysis.

## Mixture Cure Model
Assumes a proportion of individuals will never experience the event.
A logistic component determines who is cured, while uncured subjects follow an exponential failure distribution.
Mixture cure models were introduced by Farewell {ref}`Farewell1982`.

## Piecewise Exponential Model
Approximates complex hazard shapes by dividing follow-up time into intervals with constant hazard within each interval.
This yields a flexible baseline hazard while remaining computationally simple.

For additional reading on these methods please see the {doc}`bibliography`.

