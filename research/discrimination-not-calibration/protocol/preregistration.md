# Preregistration

**Study:** *The Distance Behind the C-index: Truth-Based Evaluation of Survival
Distribution Predictions*

**Protocol version:** 0.1.0
**Status:** design fixed; **not yet frozen, not yet run**

This document is written before the production Monte Carlo run and describes
what will be done. Where a decision was taken on pilot evidence, the evidence is
quoted, because "we looked and then chose" is only defensible if the looking is
on the record.

---

## 1. Question and contribution

> When conventional survival metrics continue to describe a fitted model as
> adequate, how far has the estimated individual survival distribution actually
> moved from the known data-generating distribution?

The study is not a benchmark and does not ask which estimator has the highest
C-index.

### What is not claimed

**That discrimination and calibration can disagree is established prior work,
not a finding of this study.** Austin et al. (2020) give calibration curves and
the integrated calibration index for survival models using misspecification
simulations; Sonabend et al. (2022) show discrimination scores can be inflated
by how a distribution prediction is reduced to a risk score; Birolo et al.
(2025) explicitly observe a high C-index alongside poor calibration under
non-proportional hazards; and Lillelund et al. (2025) argue the case under the
title *Stop Chasing the C-index*. The related-work section states plainly what
each establishes and what is left.

### What is claimed

Every one of those studies evaluates against **censored observations**. This
one evaluates against the **analytically known conditional survival function**,
available because each mechanism is a generator whose sampling law is written
down and validated by probability integral transform. That permits the direct
integrated squared distance between the predicted and the true individual
survival curve — a quantity no evaluation on observed data can compute, because
observed data do not contain the truth.

The contribution is therefore a **measurement, not a phenomenon**: how much
truth-error is consistent with an apparently acceptable value of each
conventional metric. The unit of interest is the metric, not the estimator; an
estimator ranking is a by-product.

## 2. Hypotheses

H1 and H2 restate an established phenomenon. They are included because the
study must reproduce it before it can quantify it, **not** as novel claims. H3
and H4 concern where it bites and how far it goes.

**H1.** Concordance and recovery of the true survival function are weakly
related across misspecified scenarios. Formally, the correlation between the
C-index and RMISE across non-null misspecified cells has absolute Spearman rank
correlation below 0.30. Pearson correlation is reported as a sensitivity
summary, not as the decision rule. The $\beta=0$ cells are a negative-control
arm rather than primary PH-violation evidence.

**H2.** There exist scenarios in which an estimator's concordance is at or above
that of a correctly specified reference while its nMISE is larger by an order of
magnitude.

**H3.** Mechanisms that break proportional hazards structurally — a
non-monotone hazard, and a survival plateau from a cured fraction — produce
larger RMISE for proportional-hazards estimators than mechanisms that only
change the baseline's shape. The primary H3 contrast is restricted to
$\beta>0$ and requires the complete structural and PH/baseline DGP sets at each
matched support point.

**H4.** Censoring degrades absolute probability recovery more than it degrades
ranking. Operationally, the contrast between 70% and 10% target censoring is
larger on mean RMISE than on mean Harrell C-index after each metric is
standardised by its across-cell standard deviation.

These are directional statements, and the study may refute any of them. The
pilot is consistent with H1 and H2 (see §7); that is a reason to run the
experiment properly, not a result.

## 3. Design

**Mechanisms (6).** `cphm` (reference: proportional hazards, exponential
baseline), `aft_weibull` (PH with a monotone parametric baseline),
`piecewise_exponential` (PH, step-function baseline), `aft_ln` (non-PH,
hazard rises then falls), `aft_log_logistic` (non-PH, unimodal hazard),
`mixture_cure` (survival plateau).

`tdcm` is excluded: its covariate changes during follow-up, so $S(t \mid X)$
given baseline covariates requires marginalising over a latent crossover time.
That is a coherent estimand, but not the one the estimators are given.
Competing-risks and multi-state generators are out of scope for a single-event
study.

**Estimators (4).** Cox proportional hazards, Weibull AFT, random survival
forest, gradient boosted survival analysis. Chosen to span assumptions —
proportionality, functional form, parametric baseline — not to lengthen a
table. No neural models: there is no methodological question here they would
answer.

**Factors.** $n \in \{250, 1000, 5000\}$; target censoring
$\in \{10\%, 30\%, 50\%, 70\%\}$; effect size $\in \{0, 0.5, 1.0\}$.

216 scenarios declared. Cells that are infeasible are dropped automatically
with the reason recorded: `mixture_cure` cannot reach 10% or 30% censoring
because a 30% cure fraction puts a floor of about 31% on the censoring rate, so
roughly 198 scenarios will run.

**Replications.** $R = 500$, from inverting
$\mathrm{MCSE} = s/\sqrt{R}$ on pilot variability at a target of $0.001$ on
MISE. This covers the 90th percentile of cells. It does **not** cover the
hardest: `aft_log_logistic` at $n = 250$ with 70% censoring needs about 5000.
Those cells will be reported with their Monte Carlo error, and differences
there described as indistinguishable rather than as findings.

## 4. Primary and secondary outcomes

**Primary:** MISE, the integrated squared error between the predicted and the
true conditional survival function over $[0, \tau]$, on an independent
evaluation sample. Raw MISE is used for within-scenario comparisons; nMISE and
RMISE are used for cross-mechanism figures, adequacy summaries and headline
claims so that different values of $\tau$ do not change the scale of the
estimand. Under the contribution stated in §1 this is not one metric among
several: it is the reference against which the conventional metrics are
assessed.

**Secondary:** MIAE and its horizon-normalised form; Harrell and Uno
concordance; time-dependent AUC; Brier score and integrated Brier score;
grouped calibration error; parameter recovery where the estimand corresponds;
fit-failure rate.

Definitions are in [estimands.md](estimands.md) and are fixed.

$\tau$ is the 80th percentile of the **latent** event-time distribution,
computed per scenario before any results are seen, and independent of the
censoring level so that varying censoring does not move the target.

## 5. Decisions taken before running, and why

**Fitting and evaluation use independent samples from the same mechanism.**
Not a convention — a correction. Scoring in-sample gave a random survival
forest a Harrell concordance of 0.78 against Cox's 0.65 on a *correctly
specified* Cox mechanism; out of sample it is 0.578 against 0.619. That gap is
overfitting, and reporting it would have manufactured the very phenomenon this
study claims to detect.

**$\tau$ and `cens_par` are resolved once per scenario and frozen.**
Recalibrating per replicate would make the mechanism random, so replicates
would not be draws from one scenario.

**The IPCW interval is resolved once per scenario and frozen.** Brier and
time-dependent AUC implementations require evaluation times inside observed
follow-up support. The production run uses the prespecified scenario-level
subgrid chosen from preparation-only matched train/evaluation support draws:
the lower bound is the maximum matched lower support and the upper bound is the
minimum matched upper support across the preparation draws; when possible, one
additional grid point is dropped from each end of that interval. A replication
that still cannot support this fixed interval records IBS and mean AUC as
unavailable rather than shortening the interval after seeing that replication.

**Cured subjects are excluded from the $\tau$ quantile.** `gen_mixture_cure`
records a cured subject's event time as `max_time * 100`, a finite sentinel
meaning "never fails". Counting it put $\tau$ at 1000 for `mixture_cure` where
every other mechanism sits between 0.96 and 5.4, so MISE was integrated over a
horizon 385 times too long. Found in the pilot; the pilot results computed
against the wrong horizon were discarded.

**Parameter recovery is reported only for Cox on the three PH mechanisms.**
Elsewhere the estimands differ and a "bias" would be the difference of two
different quantities.

**Failures are counted, never dropped.**

## 6. Analysis plan

Per cell: the mean of each metric with its MCSE and the number of replications
it rests on. Failure rates reported separately from metric means.

The adequacy region is reported as a function of $\epsilon$ over a range, not
at a single threshold. $\epsilon$ is not presented as a universal constant; its
interpretation is conditional on the loss, the mechanism, the horizon and the
application.

The headline number is the 90th percentile of RMISE conditional on bins of a
conventional metric, primarily Harrell's C-index. This answers the operational
question "how large can truth-error be among rows with similar conventional
metric values?" without turning a correlation coefficient into the claim. It is
computed over scenario-estimator cell means, and its Monte Carlo uncertainty is
propagated from the cell MCSEs with fixed-bin parametric bootstrap draws.

The executable hypothesis analysis is `scripts/analyze_hypotheses.py`, which
writes `results/processed/hypotheses.parquet`, `.json` and generated manuscript
macros. H2's correctly specified reference is `cox_ph` for `cphm`,
`aft_weibull` and `piecewise_exponential`; no exact comparator is claimed for
`aft_ln`, `aft_log_logistic` or `mixture_cure`. H3's proportional-hazards
estimators are `cox_ph` and `gradient_boosted`; its structural-violation DGPs
are `aft_ln`, `aft_log_logistic` and `mixture_cure`, compared with the
PH/baseline group `cphm`, `aft_weibull` and `piecewise_exponential` on common
$(n,\text{censoring},\beta,\text{estimator})$ support only when all three
structural and all three PH/baseline DGPs are present, with $\beta=0$ reserved
as a negative-control arm. H4 is paired on common
$(\text{DGP},n,\beta,\text{estimator})$ support for 10% versus 70% censoring;
`mixture_cure` drops out of that contrast because it has no 10% support.

Before freezing, `scripts/check_ipcw_availability.py` must pass with a minimum
availability rate of 0.95 among feasible scenarios. The implementation lives in
`scripts/audit_ipcw_availability.py`, which records the scenario-level audit.
`scripts/check_grid_convergence.py` must pass on the 10 worst
scenario-estimator cells by the latest processed cell-level loss summary, using
at least 10 matched replications, with
$|\mathrm{RMISE}_{51}-\mathrm{RMISE}_{801}| \le 0.002$ on the
survival-probability scale.

`scripts/freeze_experiment.py` consumes those two gate artifacts before writing
the production lock, verifies that both criteria passed, and records each
artifact hash and threshold/result summary inside the lock.

No inferential test is used to declare an estimator superior. Conclusions are
conditional on the mechanisms studied, and no claim of general superiority will
be made.

## 7. What the pilot showed

880 cells, 22 feasible scenarios, 10 replications. **Exploratory. Not pooled
with production results**, and `run_pilot.py` refuses to read a file containing
production rows.

- No fit or scoring failures in any cell.
- Censoring calibration held: every mechanism within 2.4 points of target.
- Spread in MISE: estimator 6.4×, mechanism 6.1×, censoring 3.2×, $n$ 1.7×. No
  factor is redundant, which is why none was dropped; two interior *levels*
  were.
- **correlation(C-index, MISE) $= -0.116$** across cells. This is retained as a
  pilot diagnostic; production H1 is defined on RMISE as above.
- On `aft_ln`, Cox at $C = 0.712$ had MISE 0.0408 while Cox at $C = 0.672$ had
  MISE 0.0025.

The last two are consistent with H1 and H2. They are pilot numbers from 10
replications and are reported here as design evidence, not as results.

## 8. Reproducibility

Seeds derive from `(master_seed, scenario_id, replication_id, stream)`, so a
run with eight workers is the same experiment as a run with one. Verified: a
4-worker and a 1-worker run of the same cells produced bit-identical seeds,
MISE, concordance, integrated Brier and calibration error.

Before the production run, `scripts/freeze_experiment.py` writes the local
artifact `protocol/experiment_lock.json`, freezing the design, the prepared
scenario values, the commit and the environment. The lock file is not committed:
it records the source commit that produced it, and committing it would change
that commit. The runner loads prepared scenarios from the lock and every
production row carries the lock hash; resumption refuses rows with a missing or
different lock hash. If package code affecting simulation changes after the
freeze, that is a new experiment version, not a continuation.

**The freeze has not happened, and no production run has been executed.**

## 9. Limitations to state in the paper

- Censoring is independent of covariates in every mechanism. Informative
  censoring is a separate study.
- Single-event only. Competing risks, recurrent events and multi-state
  processes are out of scope by design.
- Covariates are low-dimensional and independent. High dimension and
  correlation are not investigated.
- Estimators are used at default settings, without tuning. The study is about
  misspecification, not about how well a practitioner can tune a forest.
- Conclusions hold for the six mechanisms studied and do not generalise to
  arbitrary survival data.
