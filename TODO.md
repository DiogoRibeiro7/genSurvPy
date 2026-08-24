# gen_surv Roadmap

This document records what is planned, what is deliberately not planned, and the
decisions behind both.

It is organised by **capability rather than by version number**. The previous
version-labelled structure ("short-term v1.1.x", "long-term v2.x") drifted behind
the released version and implied a delivery order that never held. Items are
grouped by what they achieve and ordered by priority within each group.

Priorities favour making the simulators trustworthy over making them numerous.
`gen_surv` is more valuable as a small set of generators whose output can be
relied on than as a large set whose distributions have not been checked.

For released changes see [CHANGELOG.md](CHANGELOG.md).

## Recently completed

The 1.2.0, 1.3.0 and 2.0.0 releases were a correctness push rather than a feature
push, prompted by an external review of the R-derived generators:

- [x] Repaired the release pipeline, which had been failing on every run
- [x] Corrected the bivariate sampler, which returned `chi2(1)/2` where an
      exponential was requested and could not express negative dependence
- [x] Removed post-processing that fabricated competing-risks events
- [x] Unified the RNG contract; no simulator touches the global NumPy state
- [x] Completed the illness-death models, so `2 -> 3` transitions are emitted and
      every declared rate and coefficient affects the output
- [x] Corrected documentation that described THMM as a hidden Markov model with
      Gaussian emissions, which was never implemented

## Correctness and assurance

The highest-value work. The releases above fixed the defects that were found; the
items here are about finding the ones that have not been.

- [ ] **Distribution tests for the remaining generators.** Seven of twelve
      generators have no test that they produce their claimed distribution:
      `cphm`, the three AFT variants, `competing_risks_weibull`,
      `mixture_cure` and `piecewise_exponential`. Existing tests for these check
      shape, column names and non-negativity; `mixture_cure` additionally checks
      its cure fraction, but not its event-time distribution. The bivariate
      defect survived many releases precisely because the tests checked shapes
      rather than distributions, so this is the gap most likely to be hiding
      another one.
- [ ] **R parity fixtures.** Frozen reference outputs from the R `genSurv`
      package for the models ported from it, so divergence is detected rather
      than argued about. Column names differ by design, so parity is on values.
- [ ] **Wider property-based testing.** Hypothesis is already used in a few
      places; extend it to parameter validation and invariants across all
      generators. Complements the statistical tests rather than replacing them.
- [ ] **Centralised parameter validation.** Finite and positive checks are
      applied inconsistently, so some invalid values still reach NumPy and
      surface as confusing downstream errors.
- [ ] **Return the maturity classifier to `5 - Production/Stable`** once the
      items above are in place. It was lowered to `4 - Beta` in 1.3.0 and the
      multistate work that was its stated blocker has since landed, but shipping
      distribution tests first makes the claim defensible rather than aspirational.

## Usability

- [ ] **CLI redesign.** The unified CLI implies every registered model is
      callable, but its generic parameter plumbing does not match several model
      signatures. Move to per-model subcommands, so each exposes exactly its own
      parameters:

      ```
      gen-surv cphm ...
      gen-surv cmm ...
      gen-surv competing-risks ...
      ```

- [ ] **Scenario files.** `gen-surv simulate scenario.yaml` for reproducible,
      shareable configurations. Becomes more valuable once simulation studies are
      supported.
- [ ] **Optional dependency extras.** `matplotlib`, `lifelines`, `pyarrow` and
      `pyreadr` are mandatory today, so simulating a Weibull dataset pulls in a
      plotting stack and two file-format libraries. Move them behind `viz`, `io`
      and `sklearn` extras with `all` as a convenience.
- [ ] **Add `py.typed`** so downstream users get the annotations that are already
      written and checked.

## Repository hygiene

Small, concrete, and each one has already cost time:

- [ ] **Decide on `poetry.lock`.** It is gitignored while `ci.yml` caches on
      `hashFiles('**/poetry.lock')`, a key that matches nothing and therefore
      never varies. Either commit the lock and get a real cache, or drop the
      lock-dependent cache step.
- [ ] **Fix the `develop`/`main` divergence.** Squash-merging `develop -> main`
      leaves the squash commit outside `develop`'s history, so every subsequent
      release pull request conflicts. A conflicting pull request gets **no CI run
      at all** rather than a failing one, which reads as "still queued" and is
      easy to misdiagnose. Either use merge commits for release PRs, or reset
      `develop` to `main` after each release.
- [ ] **Remove the vestigial `[tool.semantic_release]` configuration**, which no
      workflow reads.
- [ ] **Consider migrating `[tool.poetry]` metadata to PEP 621 `[project]`.**
      Poetry 2.x warns about the current layout. Deferred because it changes
      published metadata and deserves its own release.

## Architecture

Structural work that makes the model expansion below tractable rather than
repetitive.

- [x] **`SimulationConfig` and `SimulationResult`.** `simulate()` returns the
      frame, the configuration that produced it (parameters, seed and the
      `gen_surv` version) and a `truth` mapping: coefficients actually used —
      including those drawn at random, which were previously unknowable —
      covariates, linear predictors, latent event and censoring times, cure
      status, cause-specific times, transition times, and the `tdcm` crossover
      time the frame cannot express. All twelve generators report.

      Implemented with a `ContextVar` sink rather than the tuple-returning
      wrappers first sketched here: generators keep their signatures, gain one
      `record()` call each, and the frozen baselines are byte-identical before
      and after. That mattered more than the shape of the internal API.
- [x] **Baseline hazard abstraction.** `gen_surv.baseline` defines a
      runtime-checkable `BaselineHazard` protocol -- `hazard`,
      `cumulative_hazard` and its inverse -- with frozen, self-validating
      implementations for exponential, Weibull, Gompertz, log-logistic and
      piecewise-constant forms. `gen_recurrent_events` takes either a name or an
      object, so it already samples from families it does not name. Spline
      baselines are not implemented: they need a monotone fitting step and a
      numerical inverse, which is its own piece of work.
- [ ] **Spline baseline.** A `SplineBaseline` implementing the protocol, with a
      monotone fit on the log cumulative hazard and a numerical inverse.
- [ ] **General multistate engine.** An arbitrary transition graph with
      per-transition hazards, supporting both `clock="forward"` (Markov) and
      `clock="reset"` (semi-Markov), built on the baseline hazard protocol.

      **Note on the second half of this item.** Making CMM and THMM
      configurations of the engine would change what a given seed produces:
      both draw their covariates, censoring times and latent transition times
      in a particular vectorised order, and a general engine walks the graph
      per subject instead. The frozen baselines in `tests/baselines` would all
      have to be regenerated, which is a reproducibility break for anyone who
      pinned a seed. Ship the engine as a new generator first, show it
      reproduces the CMM and THMM *distributions*, and fold the two in at the
      next major version.
- [x] **Canonical output schemas.** Documented as contracts on the output
      schemas page, and enforced: `EXPECTED_COLUMNS` in the regression suite
      pins every generator's column list, and a further test fails if a model is
      registered with the dispatcher and no frozen baseline. A layout change is
      now a failing test rather than a surprise in a release.

## Model expansion

Roughly in order of how often they are needed for realistic simulation studies:

- [x] **Recurrent events** — Andersen-Gill, and PWP in total-time and gap-time
      form, over exponential, Weibull and Gompertz baselines. Returns
      counting-process intervals with an `enum` column; distribution tests cover
      the Poisson count under a constant intensity, the mean count against the
      integrated baseline hazard for each family, the rate ratio against
      `exp(beta)`, and the gap-time scaling from the stratum effects
- [ ] **Frailty and clustered survival** — shared gamma and log-normal frailty
- [ ] **Advanced censoring** — informative, interval, left-truncated and dependent
- [ ] **Time-varying effects** — `beta(t)`, delayed effects, crossing hazards,
      change points
- [ ] **Correlated competing risks** — via copulas or shared frailty, rather than
      only independent latent times
- [ ] **Missingness and measurement error** — MCAR, MAR and MNAR mechanisms plus
      noisy covariates, for benchmark datasets that resemble real ones
- [ ] **Dataset catalog** — parameter sets that reproduce the broad shape of
      classic survival datasets, for teaching and examples

## Research framework

The direction with the most distinguishing potential: moving from "a collection
of generators" to "a tool for evaluating survival methodology".

- [ ] **Simulation study runner.** Repeated experiments across scenarios with
      bias, RMSE, coverage, power and type-I error reported across replications.
- [ ] **Fit-to-simulation adapters.** Build scenarios from models already fitted
      with `lifelines` or `scikit-survival`, so simulations can be calibrated to
      a real dataset.
- [ ] **Joint longitudinal-survival models.** Valuable, and substantially more
      complex than anything above; sensible only once the architecture work has
      landed.
- [ ] **Bayesian-friendly export** — data shaped for Stan or PyMC workflows.

## Performance

- [ ] **Vectorise the remaining subject-level loops.** `gen_thmm` still loops per
      subject. This is where the available speedup actually is.
- [ ] **Benchmark before optimising further.** The benchmark suite exists; use it
      to justify any change rather than assuming one is needed.
- [ ] **Parallel generation via `SeedSequence.spawn()`**, which gives reproducible
      independent streams. Only worth doing after vectorisation, and it depends on
      the unified RNG contract that is now in place.

## Deliberately not planned

Recorded with reasons, so they are not repeatedly reopened:

- **GPU acceleration.** It targets a bottleneck the package does not have. The
  remaining Python loops and non-vectorised inverse-transform sampling offer
  larger gains for far less complexity, and would have to be done first anyway.
- **Survival neural networks.** These belong to estimation, not simulation.
  `gen_surv` is more distinctive as a simulator that can be trusted than as
  another partial survival-analysis framework.
- **Interactive dashboards and Plotly visualisations.** Presentation work that
  competes for effort with correctness work, and is easy for users to build on
  top of the returned DataFrames.
- **An R interface.** The R ecosystem already has `genSurv`, `simsurv` and
  `coxed`. Parity fixtures against R are worth having; a wrapper is not.
- **Video tutorials and a user showcase.** Not opposed, but neither is a
  development priority.

## Decisions worth not re-litigating

- **`gen_cmm` and `gen_thmm` return different layouts on purpose.** Intervals for
  the former, a state panel for the latter, mirroring `genCMM` and `genTHMM` in R.
  Forcing them into a single schema would misrepresent how each model is analysed.
- **`seed` accepts an `int`, a `numpy.random.Generator`, or `None`,** resolved
  through one helper. Passing a generator lets several simulators share a stream.
  No simulator draws from the global NumPy state.
- **Absent outcomes are never fabricated.** If a competing cause does not occur in
  a finite sample, that is a valid result. Tests must not require every category
  to appear.
- **Column names follow package convention rather than R's**, while values follow
  R. Internal consistency across generators matters more than name-level parity,
  and parity testing works on values regardless.

## How to contribute

Contributions towards these goals are welcome. Please read
[CONTRIBUTING.md](CONTRIBUTING.md) and open an issue to discuss your approach
before submitting a pull request.

New generators are expected to arrive with tests that check their distribution,
not only the shape of the returned frame.

To propose a change to this roadmap, open an issue tagged `enhancement`.
