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
Simulates the illness-death process over states 1 (healthy), 2 (illness) and
3 (death), with Weibull transition intensities scaled by a covariate.
The three rate pairs and three coefficients map one-to-one onto the
`1 -> 2`, `1 -> 3` and `2 -> 3` transitions.
Output is in counting-process form: while a subject occupies state 1 it is at
risk of both `1 -> 2` and `1 -> 3`, so it contributes a row for each over the
same interval, and a subject that reaches state 2 contributes a further
`2 -> 3` row. Sojourn times are drawn on a reset clock, making the model
semi-Markov.
The mathematical formulation follows the counting-process approach of Andersen et al. {ref}`Andersen1993`.

## Time-Dependent Covariate Model (TDCM)
Extends the Cox model to covariates that vary during follow-up.
Covariates are simulated in a piecewise fashion with optional correlation across segments.

## Time-Homogeneous Markov Model (THMM)
Simulates a three-state model (1 healthy, 2 illness, 3 death) whose transition
intensities are constant in time, which is what makes it time-homogeneous.
Each intensity is scaled by a covariate through `rate * exp(beta * X0)`, so the
three rates and three coefficients are matched one-to-one with the
`1 -> 2`, `1 -> 3` and `2 -> 3` transitions.
Output is a panel of state observations: each subject starts in state 1 at
time 0 and contributes a further observation at each transition, or at
censoring in whichever state it then occupies.
This layout differs from the counting-process form used by CMM, matching the
distinction drawn by the R package between `genTHMM` and `genCMM`.
For background on multistate survival models see Andersen et al. {ref}`Andersen1993`.

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

