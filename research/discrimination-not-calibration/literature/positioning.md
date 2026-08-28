# Positioning and the contribution claim

**Status:** partially read.

| paper | read? |
|---|---|
| Haider et al. (2020) | **yes** — Section 3.5, Appendix B.5, Theorem B.3 |
| Lillelund et al. (2025) | **yes** — abstract, Sections 1–2 |
| Austin et al. (2020) | not yet |
| Sonabend et al. (2022) | not yet |
| Burk et al. (2026) | not yet |
| the remaining 17 | not deposited |

Claims below about the first two come from the papers themselves. Claims about
the rest come from the author's summary and are marked as such. Nothing here may
be cited in the manuscript until its source has been read.

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
- **Sonabend, Bender & Vollmer (2022)** show that distribution predictions can
  be made to look good on discrimination depending on how they are reduced to a
  risk score — "C-hacking".
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
- **Birolo et al. (2025)** explicitly observe that models with a high C-index
  can be badly calibrated under non-proportional hazards. This is the nearest
  conceptual competitor and the paper we must differentiate against most
  carefully.
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

**Antolini's time-dependent concordance** (2005) uses the predicted survival
distribution rather than a static risk score, which makes it the right
discrimination measure when the mechanism is non-proportional. We currently
report Harrell, Uno, and a concordance computed on $1 - \hat S(\tau)$; the last
is a crude stand-in for what Antolini defines properly.

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
