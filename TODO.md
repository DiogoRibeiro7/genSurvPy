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

- [x] **Distribution tests for every generator.** All twelve now have one.
      `tests/test_distributions.py` covers `cphm`, the three AFT variants, both
      competing-risks models and `mixture_cure` by probability integral
      transform: the sampled times are rearranged by the model's own cumulative
      hazard and tested against Uniform(0, 1), which states the whole
      distribution — shape, scale and covariate effect — rather than a moment or
      two. The remaining five are covered in `test_piecewise_hazards.py`,
      `test_recurrent.py`, `test_tdcm_crossover.py` and
      `test_statistical_correctness.py`.

      The gap was worth closing on the record it produced: asking
      distributional questions found the piecewise middle-interval defect and
      the `tdcm` sign error, neither of which any shape-based test could see.
      The new tests fail on a 5% error in a covariate effect and a 3% error in a
      Weibull shape, which was verified by introducing exactly those.
- [x] **R parity fixtures.** `scripts/generate_r_fixtures.R` freezes real
      output from R `genSurv` 1.0.6 into `tests/fixtures/r_parity/`, and
      `tests/test_r_parity.py` compares against it. CI never needs R.

      Parity is on **distributions, not values**: R draws from the Mersenne
      Twister and we draw from PCG64, so identical numbers are impossible
      however faithful the port. Each comparison is a statistic — a censoring
      rate, an occurrence/exposure intensity per edge, a state occupancy —
      with a three-sigma band allowing for the Monte Carlo error on both
      sides. Both sides are frozen, so the tests are deterministic. The
      measured discriminating power is recorded in the module: a 10% change in
      `cphm`'s `beta` is caught, and 15% in a `cmm` or `thmm` rate.

      Three of the four ported models agree: `cphm`, `cmm` and `thmm`. Reading
      R's source settled two apparent divergences that were not: R's `genCMM`
      labels `trans = 1` as the **1 -> 3** edge and `trans = 2` as 1 -> 2, and
      its `genTDCM` splits the risk interval at the crossover where we return
      one row per subject.

      **`tdcm` genuinely diverges, and should.** Asked for Weibull marginals
      with `dist.par = c(1, 2, 1, 2)`, R's `dgBIV` returns mean 2 and median
      2*log(2) — chi-square with two degrees of freedom, an exponential with
      mean 2 — ignoring the parameterisation it was given. Against the Weibull
      it was asked for, KS gives p = 0 for R and p = 0.87 for ours. This is the
      same defect 2.0.0 corrected for the exponential case, now recorded for
      the Weibull case, and a test pins it so nobody "fixes" our sampler into
      agreement with R.

      Three comparisons looked significant at one seed and vanished over
      eight, which is why the tests average over several.
- [x] **Wider property-based testing.** `tests/test_properties.py` drives all
      twelve generators from Hypothesis: output invariants (the column
      contract, no NaN or infinity, `status` within its declared set, no
      zero-length risk intervals, no event at time zero), the seed contract
      (the same seed gives the same frame, and an `int` agrees with the
      generator it seeds), and rejection of out-of-domain values. A test fails
      if a model is registered without a strategy, so none can escape it.

      It found two defects on its way in, both in `tdcm`.

      `validate_gen_tdcm_inputs` allowed `corr` at the endpoints — `(0, 1]` for
      Weibull and `[-1, 1]` for exponential, which the documentation promised
      too — while the Gaussian copula underneath requires strict inequalities,
      its covariance `[[1, corr], [corr, 1]]` being singular at `|corr| = 1`.
      The endpoints passed the model's own check, failed deeper in, and quoted
      a different range from a helper the caller never named.

      The second was worse. A Weibull `dist_par` shape below 1 is an exponent
      above 1 in `(-log(1 - u) / a) ** (1 / b)`, so the covariate reached the
      tens of thousands and `exp(beta[0] * z)` left the range of a float. Then
      `t = log_term / inf` is exactly 0.0 and `status = (t <= c)` reported an
      **observed event at time zero** for every subject, in a zero-length risk
      interval; with the sign of `beta[0]` flipped it underflowed instead and
      every subject came back censored. Both frames had the right columns, the
      right dtypes and no NaN, so a finiteness check passed them. `gen_tdcm`
      now raises, naming the largest covariate drawn and what to change.
- [x] **Centralised parameter validation.** Finiteness is now checked where
      positivity is. The inconsistency had a single cause: every comparison
      with NaN is false, so `value <= 0` admitted it, and `inf > 0` is true.
      `ensure_positive_sequence` had guarded against both; the scalar
      `ensure_positive` had not.

      Probing all twelve generators with NaN and infinity in every numeric
      argument found **39 that were accepted**. They came back as a frame of
      the right shape quietly full of NaN, as `OverflowError: high - low range
      exceeds valid bounds` from a uniform draw, or — for
      `gen_recurrent_events(followup_time=nan)` — as a call that never
      returned, the sampling loop comparing candidates against a bound nothing
      can exceed. `cmm` and `thmm` also checked only the *length* of `rate`, so
      a negative entry surfaced as `ValueError: scale < 0` from inside NumPy.

      All rejected now, with `tests/test_input_hardening.py` walking every
      numeric argument of every model to keep it that way.
- [x] **Returned the maturity classifier to `5 - Production/Stable`.** All
      three stated conditions are met: distribution tests and property-based
      tests for all twelve generators, and R parity fixtures for the four
      ported from R. Worth waiting for those: the
      distribution work found two more defects on its way in, which is not yet
      the profile of a package claiming stability.

## Usability

- [ ] **CLI redesign.** Every registered model is now actually callable —
      `--rate`, `--dist`, `--corr`, `--dist-par` and `--lam` were added for
      `cmm`, `thmm` and `tdcm`, which previously failed with a TypeError about
      missing positional arguments, and a parametrised test now covers all
      twelve. The plumbing is still generic, though: one `--rate` means six
      values for `cmm`, three for `thmm` and one for `recurrent_events`. Move to
      per-model subcommands, so each exposes exactly its own parameters:

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
- [x] **Added `py.typed`.** The annotations were written and mypy checked them
      on every commit, but PEP 561 tells a type checker to ignore an installed
      package's inline types without the marker, so none of it reached anyone.
      A downstream `mypy` reported `Cannot find implementation or library stub
      for module named "gen_surv"` and silently accepted
      `gen_cphm(n="not an integer", ...)`; it now reports the argument type.
      Measured by installing the built wheel into a clean environment and
      running mypy over a downstream file, before and after.

## Repository hygiene

Small, concrete, and each one has already cost time:

- [x] **Committed `poetry.lock`.** It was gitignored while five cache steps
      keyed on `hashFiles('**/poetry.lock')` — a key matching nothing, so it
      never varied and the cache never invalidated. Committing it makes CI
      resolve the same set every run and gives those keys meaning. It does not
      constrain anyone installing from PyPI, which resolves against the
      published metadata instead.

      Three things depended on the lock being tracked and had therefore never
      worked:

      - `update-poetry.yml` decided whether to open a pull request with
        `git status --porcelain poetry.lock`, which reports nothing for an
        ignored file, so the answer was always "no changes". It also ran
        `poetry update`, bumping every dependency rather than re-resolving
        after a `pyproject.toml` change, and keyed its cache on a step id that
        did not exist.
      - `auto-upgrade-pyproject.yml` verified its work with `poetry lock
        --check`, removed in Poetry 2, which the workflow installs as "latest".
      - Every cache step lacked `restore-keys`, so a lock change meant starting
        from an empty cache rather than the nearest one.

      `ci.yml` now runs `poetry check --lock`, so a `pyproject.toml` edit
      without a matching relock fails immediately instead of resolving
      differently in silence.
- [x] **Fixed the `develop`/`main` divergence.** Release pull requests use
      merge commits now, and `main` is merged back into `develop` immediately
      after each one, so the branches stay level and no release PR replays an
      earlier merge as a phantom diff.

      A second cause of "no CI run at all" turned out to be unrelated and
      worse: both workflows were scoped to `main`, so **every** pull request
      into `develop` reported no checks, conflicting or not. Tests only ran at
      the release boundary, batched, after the commits were already in. Both
      now run on `develop` as well, and the concurrency group is keyed on the
      pull request number so a PR run cannot cancel a branch run.
- [x] **Removed the vestigial `[tool.semantic_release]` configuration**, which no
      workflow read, along with the `python-semantic-release` development
      dependency that existed only to serve it.
- [x] **Migrated `[tool.poetry]` metadata to PEP 621 `[project]`.** `poetry
      check` had been reporting three deprecations: `documentation`, `scripts`
      and the license classifier. Name, version, description, authors, keywords,
      classifiers, `requires-python`, the runtime dependencies, `[project.urls]`
      and `[project.scripts]` now live under `[project]`; `[tool.poetry]` keeps
      only `packages` and the dev and docs groups, which is the layout Poetry 2
      recommends. `poetry check` is clean.

      The wheel's metadata gains `License-Expression: MIT` in place of the
      deprecated classifier and now ships `LICENSE`; the nine runtime
      constraints, the `gen_surv` entry point and the resolved dependency set
      are unchanged. Verified by installing the built wheel into a clean
      virtual environment and running the package and its console script.

      Two things read the old layout and were updated with it: `publish.yml`,
      which took the release version from `tool.poetry.version`, and
      `scripts/pyproject_updater.py`, which chose one layout and would have
      stopped seeing the runtime dependencies.

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
- [x] **General multistate engine.** `gen_multistate` walks an arbitrary
      transition graph. Each edge is a `Transition` carrying its own
      `BaselineHazard` and coefficients, both clocks are supported --
      `clock="forward"` for a Markov process, `clock="reset"` for a semi-Markov
      one -- and either canonical layout can be returned. Cycles are allowed, so
      recovery is a transition like any other, and a state with no outgoing
      edge is absorbing.

      **CMM and THMM are now configurations of it.** `gen_cmm` builds the
      illness-death graph with Weibull edges on a reset clock; `gen_thmm` builds
      it with exponential edges and asks for the panel layout. Their columns,
      dtypes and id bases are unchanged, and every distribution test still
      passes, but **a given seed produces different data**, which is why this
      landed as 3.0.0. The frozen baselines for those two models were
      regenerated; the other ten are untouched, which is the evidence the change
      is confined.

- [x] **Vectorised the multistate engine.** It advances the whole cohort a wave
      at a time: subjects sharing a state are drawn for together, so the sampling
      and the inversions are array operations and the number of Python
      iterations is the longest path any subject takes rather than the number of
      subjects. The frame is assembled from concatenated arrays, which turned
      out to cost more than the sampling once the loop was gone.

      Measured as the minimum of fifteen runs, at n=10000: `cmm` 135.7 ms to
      9.8 ms, `thmm` 165.0 ms to 9.9 ms. Against 2.1.0, before the engine
      existed, `cmm` is 1.7x slower (5.9 ms) and `thmm` 8.8x faster (87.3 ms),
      the latter because it had always looped per subject.

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

- [ ] **Vectorise the remaining subject-level loops.** `gen_thmm` no longer
      does — the multistate engine removed that loop, with the measurements in
      the architecture section above. `cphm`, `piecewise_exponential`,
      `mixture_cure`, both competing-risks models and `recurrent_events` still
      draw one subject at a time, and that is where the remaining speedup is.
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
