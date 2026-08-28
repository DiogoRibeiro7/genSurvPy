# Positioning and the contribution claim

**Status:** revised on the author's reading of the recent literature. **None of
the cited papers have been read by the agent that wrote this file** — the PDFs
are not yet in the repository-local `papers/` directory. What follows records a
decision and its rationale; it is not a literature review, and it must not be
turned into one until the papers have actually been read.

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
- **Lillelund, Qi, Greiner & Pedersen (2025)**, *Stop Chasing the C-index*, is
  close enough to our working title that "the C-index is insufficient" cannot
  be our novelty.
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

**D-calibration** (Haider et al. 2020) and **A-calibration** (Simonsen &
Waagepetersen 2025) are calibration measures designed for individual survival
*distributions* rather than for a single horizon. Our current grouped
calibration error is a decile comparison at $\tau$, which is coarser and
engages the calibration literature less directly. Both are computable from what
each replicate already produces, and neither needs another fit — so they are
cheap to add and would strengthen exactly the component the repositioning makes
central.

**Antolini's time-dependent concordance** (2005) uses the predicted survival
distribution rather than a static risk score, which makes it the right
discrimination measure when the mechanism is non-proportional. We currently
report Harrell, Uno, and a concordance computed on $1 - \hat S(\tau)$; the last
is a crude stand-in for what Antolini defines properly.

**The working title.** *Discrimination Is Not Calibration* asserts the claim
that is no longer ours. A title naming the measurement rather than the
phenomenon would match the contribution — something built around evaluation
against known truth, or the distance between an adequate-looking model and the
generating distribution. This is the author's call.

---

## How this directory is used

`papers/` at the repository root holds the PDFs. It is gitignored and must
never reach GitHub; `tests/test_papers_not_tracked.py` enforces that, including
against `git add -f` and against copying a PDF into a tracked directory.

This directory holds **our own** notes, synthesis and bibliographic metadata.
`../paper/references.bib` carries the entries as supplied, with DOIs
unverified. No extracted text from a copyrighted paper belongs here, and no
entry should be cited in the manuscript before it has been read.
