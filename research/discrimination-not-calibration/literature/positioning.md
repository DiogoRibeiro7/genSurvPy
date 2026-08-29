# Positioning and the contribution claim

**Status:** partially read.

| paper | read? |
|---|---|
| Haider et al. (2020) | **yes** — Section 3.5, Appendix B.5, Theorem B.3 |
| Lillelund et al. (2025) | **yes** — abstract, Sections 1–2 |
| Struthers & Kalbfleisch (1986) | **yes** — Sections 1–3.1, Theorem 2.1 |
| Antolini et al. (2005) | **yes** — Summary, Sections 2.2.2–2.3.1 |
| Birolo et al. (2025) | **yes** — Abstract, Sections 3.1–3.3 |
| Sonabend et al. (2022) | **yes** — Abstract, Sections 1–3.1 |
| Austin et al. (2020) | not yet |
| Burk et al. (2026) | not yet |
| Cox (1972), Grambsch & Therneau (1994), | |
| Schemper et al. (2009), Harrell et al. (1982), | deposited, not yet read |
| Graf et al. (1999), Heagerty et al. (2000) | |
| Heagerty & Zheng (2005), Gerds & Schumacher (2006), | deposited, not yet read |
| Uno et al. (2011) | |
| Royston & Parmar (2002), Ishwaran et al. (2008), | |
| Hothorn et al. (2006), Steyerberg et al. (2010), | **not deposited** |
| Simonsen & Waagepetersen (2025) | |

Claims below about the five read papers come from the papers themselves. Claims
about the rest come from the author's summary and are marked as such. Nothing
here may be cited in the manuscript until its source has been read.

---

## The claim that is no longer available

The study cannot be positioned as discovering that

$$
\text{discrimination} \neq \text{calibration}.
$$

That is established, and recently and explicitly so. On the author's reading:

- **Austin, Harrell & van Klaveren (2020)** give graphical calibration curves
  and the integrated calibration index for survival models, and use
  misspecification simulations to do it.
- **Sonabend, Bender & Vollmer (2022)** — **read**. They define three forms of
  C-hacking: **(I)** computing several concordance indices and reporting only
  the most favourable; **(II)** reporting different indices under one generic
  name; **(III)** evaluating distribution predictions with a discrimination
  measure without justifying the transformation used to get there.

  **This study was doing the third.** Harrell's concordance was computed on each
  model's *native* risk: a partial hazard for Cox, a negative expected survival
  time for the Weibull AFT, and scikit-survival's summed cumulative hazard for
  the forest and the boosted model. Three different mathematical objects
  compared with one measure, which is the comparison the paper calls virtually
  meaningless.

  Fixed by deriving the score from the predicted curve in one fixed way for
  every model — expected mortality, the summed cumulative hazard over the
  evaluation grid — so the only thing differing between estimators is the curve
  itself. The native scores are still reported, separately and under their own
  names, which the paper explicitly says is legitimate; only conflating them is
  not. On the smoke case the change is nil for the parametric models and
  −0.0067 for the forest, and it should matter more where hazards are not
  proportional.

  Two further obligations follow for the manuscript. The study reports four
  discrimination measures, so it must report **all** of them rather than the
  one that suits a conclusion (Type I), and must name each distinctly rather
  than calling any of them "the c-index" (Type II).
- **Lillelund, Qi, Greiner & Pedersen (2025)**, *Position: Stop Chasing the
  C-index when Evaluating Survival Analysis Models* — **read**. Now published
  at ICML 2026 (PMLR 306), not a preprint, which raises the bar for
  differentiation.

  It is a **position paper**: a survey of 92 methodological and application
  papers from 2023–2025 finding roughly 72% rely on metrics misaligned with
  their stated objective, plus a "ladder hypothesis" relating models, metrics
  and censoring assumptions, illustrated by controlled experiments. Its
  emphasis is that censoring assumptions are usually left implicit, and that
  metric choice must follow from the research objective.

  More differentiable than it first appeared. Their contribution is *normative
  and diagnostic* — practice is misaligned, here is a framework for aligning
  it. Ours is *quantitative* — given an apparently acceptable metric value,
  here is how far the estimated distribution has actually moved from a known
  truth. They argue the C-index is overused; we measure what its use permits.
  Their Figure 1 also shows D-calibration is among the least-used measures in
  the surveyed literature, which supports reporting it here.
- **Birolo et al. (2025)**, *Beyond Cox Models* — **read**. The nearest
  competitor, and closer than the summary suggested. They benchmark eight
  methods (four of them deep learning) on three synthetic and three clinical
  datasets, and they already hold two positions this study might have thought
  were its own:

  - they argue Harrell's C is *improper* for non-proportional models and
    advocate **Antolini's index** instead, which is exactly the substitution
    made here — so using Antolini is following their recommendation, not
    innovating;
  - they observe that "occasionally high C-index models happen to be badly
    calibrated" and recommend pairing Antolini's C with the Brier score.

  **But their evaluation is entirely on observed data.** Section 3.3 lists
  four metrics: Harrell's C at three time quartiles, Antolini's C, the Brier
  score, and time-dependent AUROC. Their synthetic datasets are *generated
  from* a known survival function — events sampled according to a survival
  function built from the features — and that function is never used to score
  anything.

  So the nearest competitor had the truth available and did not measure against
  it. That is the cleanest statement of this study's gap, and it comes from the
  competitor's own methods section rather than from an assumption about what
  they did.

  Their synthetic mechanisms are LinPH, NonLinPH and NonPH (piecewise constant
  over 16 intervals). Ours overlap on the PH/non-PH axis and add a cure
  fraction, whose survival plateau no proportional-hazards model can represent.
  Their conclusion is guidance on *which model to choose* given sample size,
  non-linearity and non-PH; ours is *how much error a metric permits*. Those
  are different questions and should be stated as such rather than contrasted
  competitively.
- **Burk et al. (2026)** set the current benchmarking context: 21 models, 34
  datasets, several discrimination, calibration and scoring metrics.

Restating any of this as a finding would be a weaker paper making a claim
someone else already owns.

---

## The claim that is available

Every study above evaluates against **censored observations**. Ours evaluates
against the **analytically known conditional survival function**:

$$
S_{\text{true}}(t \mid x) = P(T > t \mid X = x),
$$

which is available here because the mechanism is a `gen_surv` generator whose
sampling law is written down and validated. That permits the direct quantity

$$
\int_0^\tau \bigl[\hat S(t \mid x) - S_{\text{true}}(t \mid x)\bigr]^2 \, dt,
$$

which no evaluation on observed data can compute, because the observed data
never contain $S_{\text{true}}$.

The sharpened question is therefore not "do these metrics disagree?" but:

> **When conventional survival metrics continue to describe a fitted model as
> adequate, how far has the estimated individual survival distribution actually
> moved from the known data-generating distribution?**

This reframes the contribution from *a phenomenon* to *a measurement*. The
phenomenon is prior work; the calibrated ruler is ours.

**The strongest evidence that the gap is real comes from the nearest
competitor.** Birolo et al. (2025) generate synthetic survival data *from* a
known survival function and then evaluate with Harrell's C, Antolini's C, the
Brier score and time-dependent AUROC — four measures computed from observed
outcomes. The generating function is available to them throughout and is never
used to score a prediction. A study can therefore hold the truth and still not
measure against it, which is what this one does differently.

### What this changes in the paper

1. **The comparison is between metrics, not between estimators.** The unit of
   interest becomes "how much truth-error does a given value of the C-index, or
   the integrated Brier score, or the calibration error, actually permit?" An
   estimator ranking is a by-product, not the result.

2. **The primary outcome is unchanged** — MISE against the known truth — but
   its role changes. It is no longer one metric among several; it is the
   reference against which the others are assessed.

3. **A quantitative headline becomes possible.** Instead of "discrimination and
   calibration can disagree", the paper can report the *range* of truth-error
   consistent with an apparently acceptable value of each conventional metric.

4. **The related-work section must engage Birolo et al. and Lillelund et al.
   directly**, and state plainly what they establish and what is left. Anything
   less will read as an attempt to claim their result.

---

## Consequences for the design, not yet acted on

Three follow from the repositioning. All would change the frozen experiment, so
none has been applied.

**A flexible parametric comparator.** Royston & Parmar (2002) spline models sit
between the rigid parametric and the fully non-parametric estimators, which is
the interesting middle of the misspecification axis. `lifelines.CRCSplineFitter`
is available and is the same family Austin et al. used. Adding a fifth
estimator raises the production run from 432,000 to 540,000 cells.

**D-calibration** is implemented and has now been **verified against Haider et
al. (2020)**. The censoring weights match the proof of Theorem B.3 exactly: the
bucket holding $S_c$ receives $(S_c - p_k)/S_c$, every bucket entirely below it
receives $(p_{k+1} - p_k)/S_c$, uncensored subjects contribute weight one to
their own bucket, and the test is Pearson's chi-square against uniform at
$p > 0.05$. It was written from a description and turns out to agree with the
source.

Reading it surfaced a precondition the study must report. **Theorem B.3 assumes
survival curves are strictly monotonically decreasing.** Where a curve is flat,
terms in the proof fail to cancel and buckets spanning the flat region take more
than their share, inflating the statistic and over-rejecting. The random
survival forest and gradient boosting predict step functions with exactly such
flats, so part of their worse D-calibration in our results is an artefact of the
measure rather than evidence about the model. The parametric estimators are
unaffected. Censoring pushes the other way — it smooths the bucket proportions
and raises the p-value — so the test is conservative under heavy censoring,
which is also why Kaplan–Meier is only *asymptotically* D-calibrated.

**A-calibration** (Simonsen & Waagepetersen 2025) is unread and unimplemented.
Given the flat-region problem above it may be the better measure for the
step-function estimators, which is worth settling before the freeze.

**Antolini's time-dependent concordance** is implemented and has now been
**checked against the paper**. Equation 11 is

$$
C^{td} = P\bigl(\hat S(T_i \mid X_i) < \hat S(T_i \mid X_j)
\;\big|\; T_i < T_j,\; D_i = 1\bigr),
$$

and the comparability rule — subject $i$ must have failed, subject $j$ need only
have a later observed time, censored or not — was already right. Two details
were not.

**The horizon.** Section 2.3 restricts the index to $[0, \tau]$ by
*administratively censoring at $\tau$*, not by evaluating late events at the
boundary. The implementation had clipped an event at $T_i > \tau$ to the last
grid point and still counted it as an event, inventing comparisons the
definition excludes.

**Ties.** Equation 12 uses a strict inequality, so a tie contributes zero, not
the one half of Harrell's convention. This is not a technicality here, and its
effect is asymmetric across the estimators:

| estimator | tie fraction | effect of the 0.5 convention |
|---|---|---|
| `cox_ph` | 0.000 | none |
| `weibull_aft` | 0.000 | none |
| `gradient_boosted` | 0.013 | +0.007 |
| `random_survival_forest` | 0.057 | **+0.029** |

The parametric models predict smooth curves and have no ties at all; the
step-function models do, so the convention alone moves their index by up to
three hundredths. The published definition is used and the tie fraction is
reported alongside, because a reader comparing these numbers with another
implementation — most of which use 0.5 — would otherwise see the step-function
models systematically lower for a purely conventional reason. This is precisely
the class of problem Sonabend et al. (2022) name: how a distribution is turned
into a comparison changes the number.

Together with the D-calibration flat-region caveat above, **both** distributional
measures are biased against the step-function estimators, for different and
independent reasons. The paper must say so rather than let two artefacts
accumulate into an apparent finding.

**The working title.** *Discrimination Is Not Calibration* asserts the claim
that is no longer ours, and sits close to an ICML position paper that makes it.
A title naming the *measurement* would match the contribution. Candidates:

- *How Wrong Can an Adequate-Looking Survival Model Be?* — states the question
  the study answers.
- *Measuring What Survival Metrics Permit: Evaluation Against a Known
  Data-Generating Truth* — names the method.
- *The Distance Behind the C-index: Truth-Based Evaluation of Survival
  Distribution Predictions* — names the gap.

Author's call.

---

## How this directory is used

`papers/` at the repository root holds the PDFs. It is gitignored and must
never reach GitHub; `tests/test_papers_not_tracked.py` enforces that, including
against `git add -f` and against copying a PDF into a tracked directory.

This directory holds **our own** notes, synthesis and bibliographic metadata.
`../paper/references.bib` carries the entries as supplied, with DOIs
unverified. No extracted text from a copyrighted paper belongs here, and no
entry should be cited in the manuscript before it has been read.
