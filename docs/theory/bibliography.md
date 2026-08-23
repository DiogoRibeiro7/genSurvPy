# Bibliography

Sources for the models implemented in `gen_surv`.

## Foundational papers

**Cox (1972)**
Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal
Statistical Society: Series B*, 34(2), 187–220.
→ [Cox proportional hazards](../models/cphm.md)

**Kaplan and Meier (1958)**
Kaplan, E. L., & Meier, P. (1958). Nonparametric Estimation from Incomplete
Observations. *Journal of the American Statistical Association*, 53(282),
457–481.
→ the estimator behind [`plot_survival_curve`](../guides/plotting.md)

**Farewell (1982)**
Farewell, V. T. (1982). The Use of Mixture Models for the Analysis of Survival
Data with Long-Term Survivors. *Biometrics*, 38(4), 1041–1046.
→ [Mixture cure](../models/mixture-cure.md)

**Fine and Gray (1999)**
Fine, J. P., & Gray, R. J. (1999). A Proportional Hazards Model for the
Subdistribution of a Competing Risk. *Journal of the American Statistical
Association*, 94(446), 496–509.
→ [Competing risks](../models/competing-risks.md), and why subdistribution
coefficients differ from the cause-specific ones

## Books

**Andersen, Borgan, Gill and Keiding (1993)**
*Statistical Models Based on Counting Processes*. Springer.
→ the counting-process formulation used by [CMM](../models/cmm.md)

**Fleming and Harrington (1991)**
*Counting Processes and Survival Analysis*. Wiley.

**Kalbfleisch and Prentice (2002)**
*The Statistical Analysis of Failure Time Data*. Wiley.
→ AFT parameterisations, competing risks

**Klein and Moeschberger (2003)**
*Survival Analysis: Techniques for Censored and Truncated Data*. Springer.
→ censoring mechanisms

**Therneau and Grambsch (2000)**
*Modeling Survival Data: Extending the Cox Model*. Springer.
→ [time-dependent covariates](../models/tdcm.md) and the `(start, stop]` layout

**Cook and Lawless (2007)**
*The Statistical Analysis of Recurrent Events*. Springer.

**Collett (2015)**
*Modelling Survival Data in Medical Research*. CRC Press.

**Kleinbaum and Klein (2012)**
*Survival Analysis: A Self-Learning Text*. Springer.
→ a gentler entry point than the others

**Zucchini, MacDonald and Langrock (2017)**
*Hidden Markov Models for Time Series*. Chapman and Hall/CRC.
→ background on Markov chains; note that [THMM](../models/thmm.md) is an
*observed*-state model, not a hidden one

## Software

**genSurv (R)**
[cran.r-project.org/package=genSurv](https://cran.r-project.org/package=genSurv)
— the package `gen_surv` is a port of. `genCMM` and `genTHMM` are the origin of
the two illness-death layouts.

**lifelines**
[lifelines.readthedocs.io](https://lifelines.readthedocs.io) — Kaplan-Meier,
Cox and AFT fitting; a hard dependency here.

**scikit-survival**
[scikit-survival.readthedocs.io](https://scikit-survival.readthedocs.io) —
machine-learning survival models; optional, see
[Interoperability](../guides/interoperability.md).
