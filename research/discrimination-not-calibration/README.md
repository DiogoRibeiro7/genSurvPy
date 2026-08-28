# Discrimination Is Not Calibration

A controlled simulation study of survival-model misspecification.

> Can a survival model maintain apparently good discrimination while producing
> materially incorrect individual survival probabilities under model
> misspecification?

This directory contains the whole study: code, configuration, protocol,
results and manuscript. It uses [`gen_surv`](../../gen_surv) as its simulation
engine and imports it normally. **No package source is duplicated or moved
here, and nothing in `gen_surv` imports anything from this directory.** The
dependency runs one way.

The paper is a methodological study, not documentation for `gen_surv`. The
package appears in the methods and reproducibility sections because it supplies
controlled mechanisms with known latent truth, which is what makes the central
comparison possible; it is not the subject.

---

## Why known truth matters

An ordinary benchmark compares predictions against censored observed outcomes.
That conflates the error of the model with the noise of one realisation, and it
cannot separate the four things this study needs to keep apart:

| | measures | available without the truth? |
|---|---|---|
| **Discrimination** | ranking | yes |
| **Calibration** | agreement of predicted with observed risk | yes |
| **Prediction error** | squared error against the observed outcome | yes |
| **Recovery of the DGP** | distance from $\hat S_i(t)$ to $S_i(t)$ | **no** |

Because the mechanism is known here, the fourth is computable. The study's
operational question is whether the first three detect what the fourth
measures.

---

## Layout

```
config/          scenario, estimator and metric definitions (generated; see scripts/make_config.py)
src/             the study package, survival_misspec
scripts/         config generation, running, aggregation, figures
tests/           validation suite, run by the repository's normal pytest
protocol/        estimands, preregistration, experiment_lock.json
literature/      our own notes and bibliography (never source PDFs)
results/         raw replicate rows, processed tables, figures
paper/           LaTeX manuscript
```

## Running it

```bash
python scripts/make_config.py --pilot
python scripts/run_simulation.py --out results/raw/pilot.parquet
python scripts/run_pilot.py --raw results/raw/pilot.parquet
```

Production additionally requires a frozen experiment lock, and refuses to start
without matching it:

```bash
python scripts/run_simulation.py \
    --out results/raw/production.parquet \
    --lock protocol/experiment_lock.json
```

## Design decisions worth knowing before reading the code

**Fitting and evaluation use independent samples.** The first version of this
pipeline scored models on the data they were fitted to, and a random survival
forest scored a Harrell concordance of 0.78 against Cox's 0.65 on a *correctly
specified* Cox mechanism. Out of sample it scores 0.578 against 0.619. That gap
was overfitting, and reporting it would have manufactured the very phenomenon
the paper claims to find.

**`tau` and `cens_par` are resolved once per scenario and frozen.**
Recalibrating either per replicate would make the mechanism itself random, so
replicates would no longer be draws from one scenario. `tau` comes from the
*latent* event times, so it does not move when the censoring level does.

**Seeds are derived from identifiers, not from a counter.** A seed is a pure
function of `(master_seed, scenario_id, replication_id, stream)`, so parallel
and sequential runs produce the same experiment, a resumed run reproduces
exactly what an uninterrupted one would have, and adding a scenario perturbs no
other.

**Infeasible cells are reported, not approximated.** `mixture_cure` with a 30%
cure fraction cannot produce 10% censoring — a cured subject never fails — so
the censoring floor is about 31%. Those cells are dropped with the reason
stated rather than silently run at the nearest achievable rate.

**Parameter recovery is reported only where the estimand corresponds.** Cox on
the three proportional-hazards mechanisms, and nowhere else. Differencing a
Weibull AFT coefficient from a log-normal `beta` would produce a number that is
not a bias.

**Failures are results.** A replication where an estimator does not converge is
counted and reported, never dropped.

---

## `papers/` is not here, and must not be

Literature PDFs live in `<repo root>/papers/`, which is gitignored and must
never reach GitHub. Only our own notes and bibliographic metadata are committed,
under `literature/`. `tests/test_papers_not_tracked.py` in the repository root
test suite fails the build if anything below `papers/` is tracked, or if any PDF
is tracked anywhere.

---

## Reproducibility

For every number in the paper it must be possible to reconstruct

```
result -> replication -> seed -> scenario -> DGP parameters
       -> estimator -> metric -> gen_surv version -> git commit
```

Every result row carries its seeds, scenario and estimator hashes, the study
hash, the `gen_surv` version and the commit. `protocol/experiment_lock.json`
freezes the design and the environment, and the runner refuses to proceed
against a mismatched lock.

Because `gen_surv` lives in this same repository, recording a version string is
not sufficient — the generators can change while the version does not. The lock
records the **commit**, and `capture_provenance` additionally cross-checks the
installed package metadata against `pyproject.toml`: an editable install once
served 3.1.0 source while reporting version 2.0.1, which would have mislabelled
every result in the study.
