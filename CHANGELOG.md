# CHANGELOG

## Unreleased

## v3.1.2 (2026-08-30)

Final pre-freeze corrections for the discrimination-not-calibration research
study.

### Research

- Tightened the executable H1 and H3 estimands so the primary PH-violation
  analyses exclude the null-effect arm; `effect_size = 0` is now treated as a
  negative-control arm.
- Required complete common support for H3: each matched support point must
  contain all structural DGPs and all PH/baseline DGPs before contributing to
  the primary contrast.
- Corrected the headline bootstrap uncertainty: the reported quantile standard
  error is now the bootstrap standard deviation, with bootstrap Monte Carlo
  error stored separately.
- Added bootstrap SE and percentile intervals for H1-H4 and propagated them to
  generated manuscript macros and Table 6.
- Made the IPCW availability gate audit the production evaluation sample size
  instead of the scenario training size.
- Made experiment freezing consume the IPCW and grid-convergence gate
  artifacts, verify that both passed, and embed their hashes and result
  summaries in the lock.
- Included full environment metadata and gate evidence in the experiment lock
  hash, and made strict verification reject locks created from dirty source.

## v3.1.1 (2026-08-29)

R parity fixtures, and the maturity classifier returned to
`5 - Production/Stable`.

### Bug Fixes

- **Multi-state generators now reject underflowed observed transitions.** A
  transition with `start == stop` is a zero-exposure counting-process interval,
  so it is rejected instead of emitted.

### Research

- Added the truth-based survival-distribution evaluation study under
  `research/discrimination-not-calibration/`, with deterministic paired
  simulations, known-truth metrics, provenance capture, pre-freeze safeguards
  and manuscript-generation scripts.
- Corrected pre-freeze study issues found during review: native-time
  distribution metric evaluation for step-function predictions, exact
  log-logistic truth for the generator's clipped law, paired Monte Carlo
  contrasts, corrected second-stage MCSE propagation, append-only raw result
  shards, bounded parallel submission and removal of tracked pilot result
  tables.

### Dependencies

- Raised dependency floors for `scipy`, `pyarrow`, `scikit-survival`, `pytest`,
  `pytest-benchmark`, `black`, `isort`, `invoke`, `pre-commit` and
  `mkdocstrings`.

### Tests

- **R parity fixtures.** `scripts/generate_r_fixtures.R` freezes real output
  from R `genSurv` 1.0.6 into `tests/fixtures/r_parity/`, and
  `tests/test_r_parity.py` compares against it. CI never needs R installed.

  Parity is on distributions, not values: R draws from the Mersenne Twister and
  we draw from PCG64, so identical numbers are impossible however faithful the
  port. Each comparison is a statistic with a three-sigma band allowing for the
  Monte Carlo error on both sides, and both sides are frozen, so the tests are
  deterministic. `cphm`, `cmm` and `thmm` agree with R.

- **`tdcm`'s divergence from R is now pinned as intended behaviour.** Asked for
  Weibull marginals with `dist.par = c(1, 2, 1, 2)`, R's `dgBIV` returns mean 2
  and median `2*log(2)` — chi-square with two degrees of freedom — ignoring the
  parameterisation it was given. Against the Weibull it was asked for, a KS test
  gives `p = 0` for R and `p = 0.87` for ours. This is the same defect 2.0.0
  corrected for the exponential case; a test now records it for the Weibull case
  so our sampler is not "fixed" into agreement with R.

### Packaging

- **`Development Status :: 5 - Production/Stable`**, replacing `4 - Beta`. The
  three conditions the roadmap set are met: distribution tests and
  property-based tests across all twelve generators, and R parity fixtures for
  the four ported from R.

## v3.1.0 (2026-08-28)

Property-based tests across all twelve generators, and the two `tdcm` defects
they found.

### Behaviour changes

- **`gen_tdcm` raises when `exp(beta[0] * z)` leaves the range of a float**
  instead of returning a frame. A Weibull `dist_par` shape below 1 is an
  exponent above 1 in `(-log(1 - u) / a) ** (1 / b)`, so the covariate reached
  the tens of thousands and the linear predictor overflowed. `t = log_term /
  inf` is exactly 0.0, and `status = (t <= c)` then reported an **observed
  event at time zero** for every subject, in a zero-length risk interval; with
  the sign of `beta[0]` flipped it underflowed instead and every subject came
  back censored. Both frames had the right columns, the right dtypes and no
  NaN. No correct call changes behaviour — the affected parameter combinations
  never produced usable data.
- **`gen_tdcm` rejects `corr` at the endpoints.** `validate_gen_tdcm_inputs`
  allowed `(0, 1]` for Weibull and `[-1, 1]` for exponential, and the
  documentation promised the same, but the Gaussian copula underneath needs
  strict inequalities: its covariance `[[1, corr], [corr, 1]]` is singular at
  `|corr| = 1`. Those values already failed — deeper in, reporting a different
  range from a helper the caller never named. They now fail at the model's own
  boundary, quoting the range it actually enforces.

### Tests

- **`tests/test_properties.py`** drives every generator from Hypothesis:
  output invariants (the column contract, no NaN or infinity, `status` within
  its declared set, no zero-length risk intervals, no event at time zero), the
  seed contract (the same seed gives the same frame, and an `int` agrees with
  the generator it seeds), and rejection of out-of-domain values. A test fails
  if a model is registered without a strategy.

### Documentation

- The `corr` range on the TDCM page and in the `gen_tdcm` docstring now match
  what is enforced.

## v3.0.0 (2026-08-25)

A general multistate engine, with the two illness-death models rebuilt on top of
it. The engine is the reason for the major version: **`gen_cmm` and `gen_thmm`
produce different data for a given seed** than they did in 2.1.0.

### Breaking Changes
- **`gen_cmm` and `gen_thmm` no longer reproduce their 2.1.0 output for a given
  seed.** Both are now configurations of `gen_multistate`, which draws one
  candidate per outgoing edge per visit where the old implementations drew all
  three latent times up front. Their columns, dtypes and id bases are unchanged
  — `thmm` still numbers subjects from 1 — and every distribution test passes
  untouched, so analysis code keeps working. Only the numbers move. Pin the
  version alongside the seed if you need to reproduce earlier results.
- **NaN and infinity are rejected wherever a number is expected.** Calls that
  previously returned a frame quietly full of NaN, raised an unrelated
  `OverflowError` from NumPy, or — for
  `gen_recurrent_events(followup_time=nan)` — never returned at all, now raise
  a `ValidationError` naming the argument. Thirty-nine arguments across the
  twelve generators were affected. A NaN among `betas` was previously reported
  as `NumericSequenceError` and is now a `ParameterError`, since NaN is
  numeric and that error named the wrong problem.

### Features
- **`gen_multistate`**, an engine over an arbitrary transition graph. Each edge
  is a `Transition` carrying its own `BaselineHazard` and coefficients, so the
  intensity of the `i -> j` transition is `h0_ij(t) * exp(X'beta_ij)`. Both
  clocks are supported: `clock="forward"` measures the hazard from entry to the
  study, making the process Markov, and `clock="reset"` restarts it at each
  state, making it semi-Markov. Either canonical layout can be returned. A
  state with no outgoing edge is absorbing, and cycles are allowed, so recovery
  is a transition like any other.

  It is not reachable through `generate()`: a graph is a list of objects rather
  than a set of scalars, so there is no string form and no command-line
  equivalent.
- **`py.typed`.** Every public function was annotated and mypy checked them on
  every commit, but PEP 561 tells a type checker to ignore an installed
  package's inline types without the marker, so none of it reached anyone. A
  downstream `mypy` reported `Cannot find implementation or library stub for
  module named "gen_surv"` and accepted `gen_cphm(n="not an integer", ...)`
  without comment.
- `ensure_finite`, for arguments with no sign constraint. `cphm`'s `beta` is a
  log hazard ratio, so no positivity check reached it and nothing had been
  validating it at all.

### Bug Fixes
- `cmm` and `thmm` checked only the *length* of `rate`, never its contents, so
  a negative entry surfaced as `ValueError: scale < 0` from inside NumPy.

### Performance
- The engine advances the whole cohort a wave at a time: subjects sharing a
  state are drawn for together, so the sampling and the hazard inversions are
  array operations. At ten thousand subjects, `thmm` is **8.8x faster** than in
  2.1.0 — it had always looped per subject — while `cmm`, which had been
  vectorised, is 1.7x slower, the price of generality.

### Testing
- Distribution tests for the engine: each edge's intensity recovered by
  occurrence over exposure, the sojourn in a state exponential in the summed
  intensity, the destination share following the competing intensities, the two
  clocks identical for a constant hazard and sharply different for a rising one,
  cyclic graphs and absorbing states.
- `tests/test_input_hardening.py` walks every numeric argument of every model,
  scalar and sequence, and requires a `ValidationError`.
- `tests/test_packaging.py` checks the `py.typed` marker on the imported
  package, and that every public parameter and return value really is
  annotated: a marker promising types that were not there would be worse than
  no marker.

## v2.1.0 (2026-08-24)

A twelfth model, the configuration and ground truth behind every dataset, a
baseline hazard abstraction, and two sampler corrections. The documentation is
rebuilt from scratch on MkDocs; the API reference on the published site had
been empty since it was written.

**Two generators produce different data for the same seed** than they did in
2.0.1: `piecewise_exponential` with two or more breakpoints, and `tdcm`. Both
were producing wrong data, described below. Pin the version alongside the seed
if you need to reproduce earlier results.

### Features
- **`gen_recurrent_events`**, a twelfth model, for events that repeat within a
  subject. Three processes matching the models the data is analysed with:
  `ag` (Andersen-Gill), and `pwp_tt` and `pwp_gt` (Prentice-Williams-Peterson
  in total and gap time, the latter resetting the clock at each event).
  Exponential, Weibull and Gompertz baselines. Returns counting-process
  intervals with an `enum` column.
- **`simulate()`**, returning a `SimulationResult`: the frame, the
  `SimulationConfig` that produced it — parameters, seed and the `gen_surv`
  version — and a `truth` mapping of what a real dataset could never contain.
  Coefficients actually used, covariates, linear predictors, latent event and
  censoring times, cure status, cause-specific and transition times, and the
  `tdcm` crossover time the frame cannot express. All twelve generators report.

  Most useful where several generators **draw their coefficients when the
  caller omits them**: there was previously no way to learn what they were,
  which quietly made those datasets useless for validating an estimator.
- **`gen_surv.baseline`**, a `BaselineHazard` protocol with `hazard`,
  `cumulative_hazard` and its inverse, implemented for exponential, Weibull,
  Gompertz, log-logistic and piecewise-constant hazards. A generator written
  against it accepts any shape: `gen_recurrent_events` takes a name or an
  object, so it already samples from families it does not name.
- The CLI can reach every registered model. `cmm`, `thmm` and `tdcm` had no way
  to pass their parameters and failed with a `TypeError`, while being
  advertised as valid values of `MODEL`. Adds `--rate`, `--dist`, `--corr`,
  `--dist-par` and `--lam`.

### Bug Fixes
- **`gen_piecewise_exponential` drew middle-interval events at the wrong
  hazard.** The inversion assigned the event time and broke out of the loop,
  but the trailing "no event yet" branch ran anyway and overwrote it using the
  *last* rate. With breakpoints `[1, 3]` and rates `[0.5, 2.0, 0.2]`, the
  hazard measured on `[1, 3)` was 0.201 against a declared 2.0. Only
  specifications with two or more breakpoints were affected.
- **`gen_tdcm` had a sign error in its post-crossover inversion**, placing
  events drawn after the covariate switch *before* it and, for a large enough
  `beta[1]`, at negative times: 6886 of 50000 subjects at `beta[1] = 1.0`. The
  hazard ratio across the switch measured 4.58 where `exp(beta[1])` was 2.0.
- **`tdcov` described the wrong interval.** It was set from the branch the
  event time was drawn on, so a subject censored before its crossover was
  recorded as having switched though its covariate never did while observed. It
  is now whether the crossover was reached by the observed exit.
- **`summarize_survival_dataset` crashed on Windows.** Its verbose report, the
  default, printed check and cross marks that a console on a legacy code page
  cannot encode, raising `UnicodeEncodeError` before printing anything.
- **`GenSurvDataGenerator` was not scikit-learn compatible.** `get_params`
  reported only `model` and `return_type`, so `set_params` raised on any model
  argument and `clone` — used internally by pipelines, `GridSearchCV` and
  `cross_val_score` — silently dropped every parameter, producing an estimator
  that failed on first use.

### Documentation
- Rebuilt on **MkDocs** with Material and mkdocstrings, replacing Sphinx.
  `docs/source/api/index.md` had been written in mkdocstrings syntax, which
  Sphinx rendered as literal YAML, so **the published API reference contained
  no signatures and no docstrings at all**. Read the Docs is retired; the
  GitHub Pages site, built from the release tag, is the single home.
- Rewritten rather than ported: per-model pages with parameters, mathematics, a
  worked example and a check that the parameters can be recovered; guides for
  baselines, ground truth, censoring, covariates, summaries, plotting, export,
  interoperability and the CLI; a full API reference over every public module.
- Three docstrings in `gen_surv/aft.py` indented their `Returns` and `Examples`
  sections six spaces instead of four, so numpydoc never parsed them and
  neither rendered.
- The example scripts and Binder notebooks are repaired. Two passed a `qmat`
  argument removed long ago, one of them describing the Gaussian-emission
  hidden Markov model that was never implemented; two passed a deprecated third
  coefficient to `gen_tdcm`; and all three notebooks called `np.random.seed`,
  which no generator reads, so they were not reproducible.

### Testing
- The frozen-output regression suite **had been inert**: `tests/baselines` was
  empty, so every case hit a `pytest.skip` and the run reported success while
  comparing against nothing, and it covered four of twelve generators.
  Baselines are committed for all twelve, a missing one now fails, and the
  tolerance is tightened from `1e-6` to `1e-12`.
- **Distribution tests for every generator**, by probability integral
  transform, closing the roadmap's highest-priority gap.
- The documentation's examples are executed and their pasted output compared
  against what they print, and the example scripts and notebooks are run.

### Packaging
- Metadata migrated to PEP 621 `[project]`, clearing three Poetry deprecations.
  The wheel gains `License-Expression: MIT` and ships `LICENSE`.
- `poetry.lock` is committed, which makes CI reproducible and gives five cache
  keys that had never varied something to hash. Three workflows depended on it
  being tracked and had therefore never done anything.
- Removes the `[tool.semantic_release]` configuration no workflow read, and the
  `python-semantic-release` dependency that served it.

## v2.0.1 (2026-08-23)

No library changes: `gen_surv` behaves exactly as it does in v2.0.0. This release
publishes the documentation site and relaxes dependency ranges that were pinned
to a single major version.

### Continuous Integration
- The Sphinx documentation is now published to GitHub Pages at
  <https://diogoribeiro7.github.io/genSurvPy/>. The site is built when a release
  is published, from the tag that was uploaded to PyPI, so it always documents
  the released version rather than unreleased work on `develop`.

### Documentation
- Fixed the GitHub Pages Sphinx configuration. It inherited `html_static_path`
  from `docs/source/conf.py`, but Sphinx resolves that against the configuration
  directory, so `_static` did not exist and `custom.css` was never copied into
  the built site. It also pointed `html_extra_path` at a `.nojekyll` file that
  does not exist, which `sphinx.ext.githubpages` already writes. The site base
  URL now comes from the workflow rather than being hardcoded.

### Misc
- `pyarrow` accepts `>=21,<26` instead of `^21.0.0`, so it no longer holds
  installations back to the 21.x series.
- Widened the development and documentation dependency ranges for `pytest`,
  `invoke`, `black`, `isort`, `flake8`, `scikit-survival`, `myst-parser` and
  `sphinx-design`.

## v2.0.0 (2026-08-10)

Completes the illness-death models. `gen_cmm` and `gen_thmm` previously reported
only a subject's first transition, which left the `2 -> 3` transition entirely
absent from their output and several declared parameters with no effect. Both now
emit the full trajectory, so **their returned shape changes** and the major
version is incremented.

### Breaking Changes
- **`gen_cmm` now returns counting-process records** with columns `id`, `start`,
  `stop`, `from_state`, `to_state`, `status`, `X0`, replacing the previous
  `id`, `start`, `stop`, `status`, `X0`, `transition` frame. Subjects contribute
  two or three rows rather than one: while in state 1 a subject is at risk of
  both `1 -> 2` and `1 -> 3`, so it gets a row for each over the same interval
  with `status` marking whichever occurred, and a subject reaching state 2 gets a
  further `2 -> 3` row.
- The `transition` column is gone. It encoded the destination as an integer, and
  in the opposite sense to the R package's `trans` codes, which made it easy to
  misread. `from_state` and `to_state` state the transition explicitly.
- **`gen_thmm` now returns the full state trajectory.** Columns are unchanged
  (`id`, `time`, `state`, `X0`) but subjects contribute two or three rows instead
  of one: an entry observation in state 1 at time 0, then one observation per
  transition, or an observation in the occupied state at the censoring time.
- Any code that assumed one row per subject from either generator needs to group
  by `id`.

### Bug Fixes
- **`gen_cmm` ignored `rate[4]`, `rate[5]` and `beta[2]`.** It drew the `2 -> 3`
  sojourn time and discarded it, so a third of its declared parameters could be
  changed with no effect on the output whatsoever. All six rates and all three
  coefficients now reach the result.
- **`gen_thmm` ignored `rate[2]` and `beta[2]`** for the same reason.
- `gen_cmm` now resolves its seed through the shared RNG helper, so it accepts a
  `numpy.random.Generator` as well as an `int`, consistent with the other
  generators.
- Tie handling in both generators now matches the R implementation, which treats
  a censoring time equal to the first transition time as an event.

### Documentation
- The CMM and THMM sections of the algorithms and theory pages now describe the
  emitted layouts, including why the two differ: `gen_cmm` returns transition
  intervals and `gen_thmm` returns states observed at times, mirroring `genCMM`
  and `genTHMM` in the R package. The 1.3.0 note recording the missing
  trajectory as a known limitation has been removed, since it is now fixed.

### Testing
- Added `tests/test_multistate_schema.py` with 21 structural tests: row counts
  per subject, both competing transitions being at risk over a shared interval,
  the `2 -> 3` row appearing exactly when illness was observed, the reset clock
  on entry to state 2, monotone trajectories, death being terminal, and direct
  guards that every rate and coefficient influences the output. The
  parameter-influence tests were confirmed to fail against the 1.3.0 code.
- Replaced the two `gen_cmm` snapshot tests, which asserted exact values of the
  old one-row-per-subject frame, with reproducibility and schema tests.

## v1.3.0 (2026-08-10)

A scientific-correctness release. Three of the fixes below change the numbers the
simulators produce, so results generated with 1.2.0 or earlier are not comparable
with results from this release.

### Bug Fixes
- **The bivariate sampler produced the wrong distribution entirely.** It mapped
  correlated normals to uniforms with `u = 1 - exp(-z**2 / 2)`, which is the
  chi-squared(2) CDF applied to a chi-squared(1) variable. The composed transform
  reduced to `z**2 / (2 * lambda)`, so a requested Exponential(1) marginal was
  really `chi2(1) / 2` with mean 0.5 instead of 1.0. Replaced with the normal CDF,
  making this a correctly specified Gaussian copula with exact marginals.
- **Negative dependence was unreachable in the bivariate sampler.** Squaring the
  normals mapped `+r` and `-r` onto the same positive dependence, so a requested
  correlation of `-0.8` produced roughly `+0.64`. The sign is now preserved.
- **`gen_tdcm` was affected by both of the above**, since it draws its covariates
  from that sampler.
- **The competing-risks generators fabricated events.** When fewer than two
  distinct statuses appeared in a sample, both generators overwrote `status[0]`
  and `status[1]` with event labels, attaching events to subjects whose event
  times had not occurred. A cause that is absent from a finite sample is a valid
  stochastic outcome, so this post-processing has been removed.
- **`gen_tdcm` rejected its own documented signature.** The docstring specified
  two coefficients and the model uses two, but validation required three and
  silently ignored the third, so the documented call raised `LengthError`. Two are
  now accepted; three still work but emit a `DeprecationWarning`.

### Breaking Changes
- Event times, covariates and statuses differ from 1.2.0 for `gen_tdcm`,
  `sample_bivariate_distribution`, `gen_competing_risks` and
  `gen_competing_risks_weibull`. This is the point of the release, but it does
  mean any stored 1.2.0 output should be regenerated.
- `scipy` is now a declared runtime dependency. It was already installed as a
  transitive dependency of `lifelines`, so this should not change resolution.
- The PyPI maturity classifier moves from `5 - Production/Stable` to
  `4 - Beta`. A package that has just corrected the marginal distribution and
  the dependence sign of one of its core samplers is not accurately described as
  production-stable, and known correctness gaps remain: CMM and THMM report only
  the first transition rather than a full trajectory, and the CLI cannot drive
  every registered generator. The classifier is intended to return to
  `5 - Production/Stable` once the multistate output schema lands.

### Features
- Unified the RNG contract. `sample_bivariate_distribution`, `gen_tdcm` and
  `gen_thmm` drew from the global NumPy random state and could not be seeded;
  `gen_thmm` had no `seed` parameter at all. All three now accept
  `seed`, which may be an `int`, a `numpy.random.Generator` for sharing one
  stream across simulators, or `None`. No simulator touches the global state.
- Censoring draws in `gen_tdcm` and `gen_thmm` now share the caller's generator
  rather than creating an unseeded one, so a single seed reproduces a whole
  dataset.

### Documentation
- **THMM was documented as a Hidden Markov Model, which it is not.** The name
  means Time-Homogeneous Markov Model. The docs additionally described latent
  states with Gaussian emissions, none of which exists in the implementation.
  Rewritten to describe the three-state model with constant transition
  intensities that the code actually simulates, and re-cited to Andersen et al.
  instead of an HMM textbook. The known limitation that only the first transition
  is emitted is now stated explicitly.

### Testing
- Added `tests/test_statistical_correctness.py`: Kolmogorov-Smirnov tests for the
  exponential and Weibull marginals, moment checks, a dependence-sign test, a
  monotonicity test, a Spearman check against the Gaussian copula identity,
  no-fabrication tests for competing risks, and seed-reproducibility plus
  global-state-independence tests for every affected generator. Each was
  confirmed to fail against the 1.2.0 code.
- Replaced two tests that asserted the fabricated competing-risks statuses as
  required behaviour, and removed a property-based assertion that every sample
  must contain at least two distinct statuses, which is not a property the model
  guarantees.

## v1.2.0 (2026-08-10)

### Breaking Changes
- Python 3.10 is no longer supported; the minimum supported version is now 3.11.
  This is required by the current `numpy` (>=2.3) and `lifelines` (>=0.30) releases,
  neither of which ships for 3.10.
- Removed the `dev` extra. It declared `Provides-Extra: dev` with no dependencies
  behind it, so `pip install gen-surv[dev]` never actually installed anything.
  Use `poetry install --with dev` for development, or `pip install scikit-survival`
  for the optional scikit-survival integration.

### Features
- Added official support for Python 3.13; CI now tests 3.11, 3.12 and 3.13.

### Bug Fixes
- Raised the `lifelines` floor to 0.30.3. Earlier releases crash in
  `add_at_risk_counts` under numpy 2.x with
  `TypeError: only 0-dimensional arrays can be converted to Python scalars`,
  which broke every survival-curve and covariate-effect plot.
- Made optional dependencies lazy, normalized identifier handling and stabilized
  the test suite (#116).

### Continuous Integration
- Fixed dependency resolution, which failed for every job because the declared
  Python floor (3.10) was incompatible with the pinned `numpy` and `lifelines`
  constraints.
- Dropped the `scripts/check_version_match.py` steps and the `bump-version`
  workflow. The script had been deleted in #86/#90, so all three workflows
  referencing it failed. Release tags are now created manually.

### Misc
- Added Zenodo metadata.
- Updated dependency constraints (`numpy`, `pandas`, `matplotlib`, `pyarrow`,
  `typer`, `click`) and refreshed pinned GitHub Actions.

## v1.0.9 (2025-08-02)

### Features
- export datasets to RDS files
- test workflow runs on a Python version matrix
- scikit-learn compatible data generator
- compatibility helpers for lifelines and scikit-survival

### Documentation
- updated usage examples and tutorials
- document optional scikit-survival dependency throughout the docs

### Continuous Integration
- auto-tag releases using the version check script

### Misc
- README quick example uses `covariate_range`

## v1.0.8 (2025-07-30)

### Documentation
- ensure absolute path resolution in `conf.py`
- drop unsupported theme option
- define bibliography anchors and headings
- fix tutorial links to non-existing docs
- add additional references to the bibliography

### Testing
- add CLI integration test
- expand piecewise generator test coverage

### Misc
- remove fix_recommendations.md



## v1.0.0 (2025-06-06)

### Misc

- Align pyproject version with GitHub tag
- Add project Code of Conduct

## v0.7.1 (2025-04-13)

### Bug Fixes

- Fix import
  ([`3cdb59a`](https://github.com/DiogoRibeiro7/genSurvPy/commit/3cdb59acda5e60328d1d9abd57ee49252f3044fe))

### Chores

- Add tasks.py with Invoke CLI for project automation
  ([`87868b8`](https://github.com/DiogoRibeiro7/genSurvPy/commit/87868b86aec04a0e9254cee78036ac20056731e7))

- Badges
  ([`79c1d0d`](https://github.com/DiogoRibeiro7/genSurvPy/commit/79c1d0de79bfc20cfa92da6dda5eaed192bf7dc4))

- Docs
  ([`4b2704e`](https://github.com/DiogoRibeiro7/genSurvPy/commit/4b2704e7179396a1d41c60f914a327e81bd45f6b))

- Update pyproject
  ([`127d1f6`](https://github.com/DiogoRibeiro7/genSurvPy/commit/127d1f6dff84056e859d9b92e87403e34816853e))

### Documentation

- Add mathematical foundations page for all survival models
  ([`8472ff1`](https://github.com/DiogoRibeiro7/genSurvPy/commit/8472ff181ee3b882a44bc2c48c094de4db4a70fb))

- Add roadmap for advanced survival models and extend TODO
  ([`1990109`](https://github.com/DiogoRibeiro7/genSurvPy/commit/1990109f3ea0f7a4351cfdcf8880dd87108c8181))

- Add usage examples for all models in index.md
  ([`57c5fb0`](https://github.com/DiogoRibeiro7/genSurvPy/commit/57c5fb025b2cfe2eaae613e064bcd17c900ee4fc))

- Fix version number in pyproject
  ([`6cff868`](https://github.com/DiogoRibeiro7/genSurvPy/commit/6cff868bb504256918eecfad9d5d0bad3bcc97bb))


## v0.7.0 (2025-04-12)

### Chores

- Update documentation
  ([`d34e32f`](https://github.com/DiogoRibeiro7/genSurvPy/commit/d34e32f9e2181f3ed5bf23fbee60a6a0a977c738))

### Features

- **docs**: Document generic interface `generate()` and update examples
  ([`55e22a4`](https://github.com/DiogoRibeiro7/genSurvPy/commit/55e22a4a9ad4ac1af48f0fbb176c9c474a590182))


## v0.6.1 (2025-04-12)

### Bug Fixes

- Fix pyproject toml
  ([`a06f58a`](https://github.com/DiogoRibeiro7/genSurvPy/commit/a06f58aded7336b3a14d2f48db2efd160ab31323))

### Chores

- Fix readme
  ([`f4897cd`](https://github.com/DiogoRibeiro7/genSurvPy/commit/f4897cdc19cf0a43f0fb6490ff068b55c436a2e2))

### Documentation

- Docs
  ([`75417f1`](https://github.com/DiogoRibeiro7/genSurvPy/commit/75417f184dd067f0bd8bd0b7351eb3b0d1f0d336))


## v0.6.0 (2025-04-12)


## v0.5.0 (2025-04-12)

### Features

- Add documentation to readthedocs
  ([`1b00d74`](https://github.com/DiogoRibeiro7/genSurvPy/commit/1b00d740a1613e42138510dfa01331074bd97a22))


## v0.4.0 (2025-04-12)

### Bug Fixes

- Fix codecoverage
  ([`16bc525`](https://github.com/DiogoRibeiro7/genSurvPy/commit/16bc525138e6a2b4370f951cab896ae8476b1775))

### Chores

- Delete file
  ([`ba2043e`](https://github.com/DiogoRibeiro7/genSurvPy/commit/ba2043eb3b891c6fe8d062e655002573cfb2e6fa))

- Fix documentation
  ([`4231eca`](https://github.com/DiogoRibeiro7/genSurvPy/commit/4231eca916b50eabaa0e79472ce4481910d02a96))

### Features

- Add documentation to readthedocs
  ([`d1a0d29`](https://github.com/DiogoRibeiro7/genSurvPy/commit/d1a0d29e38f61a4366b8e845415fc60bee3b7ca2))

- Add documentation to readthedocs
  ([`78eb448`](https://github.com/DiogoRibeiro7/genSurvPy/commit/78eb4485ff801ba9e951ba5f2a1e51e6c3d1d468))


## v0.3.1 (2025-04-12)

### Bug Fixes

- Bump version
  ([`5cc649d`](https://github.com/DiogoRibeiro7/genSurvPy/commit/5cc649da03bbc57fbaf80494e63543012e1e849d))

### Chores

- Bump version
  ([`2e269f5`](https://github.com/DiogoRibeiro7/genSurvPy/commit/2e269f50737a5b497718f55e1e42a1501e0d398f))

- Bump version
  ([`081ce3d`](https://github.com/DiogoRibeiro7/genSurvPy/commit/081ce3db093410ea96f8d594201d26f81f6b35ec))

### Features

- Add documentation to readthedocs
  ([`c33f666`](https://github.com/DiogoRibeiro7/genSurvPy/commit/c33f666295dd7a779065249bd61bf28631519e79))


## v0.3.0 (2025-04-12)

### Bug Fixes

- Fix git hub actions
  ([`c9559ba`](https://github.com/DiogoRibeiro7/genSurvPy/commit/c9559ba319684e5b9de59bfa7ad23e42ee6f8bf9))

### Chores

- Bump version
  ([`e111826`](https://github.com/DiogoRibeiro7/genSurvPy/commit/e11182609d99094d8b02fa2e142ac983f42dc5b3))


## v0.2.1 (2025-04-12)


## v0.2.0 (2025-04-12)


## v0.1.0 (2025-04-12)

### Bug Fixes

- Add name to pyproject
  ([`6a3a8f3`](https://github.com/DiogoRibeiro7/genSurvPy/commit/6a3a8f3a82b6017f1903f23b27d02233a05e0763))

- Fix mixing tags creation
  ([`93ac8a0`](https://github.com/DiogoRibeiro7/genSurvPy/commit/93ac8a05ac8b825b2c5066a0f7eb4e8b7463fb06))

- Fix pyproject toml
  ([`8537f63`](https://github.com/DiogoRibeiro7/genSurvPy/commit/8537f63d01249e8db29e9e7353831d0b894bbc3d))

- Fix recommended by copilot
  ([`f188f76`](https://github.com/DiogoRibeiro7/genSurvPy/commit/f188f76fa858f1753ad1b153f61df94a241f0831))

- Fix semantic release
  ([`3d18b02`](https://github.com/DiogoRibeiro7/genSurvPy/commit/3d18b02fb617dddade3fb6862d5647143e52a3a9))

- Fix version bump
  ([`e5114de`](https://github.com/DiogoRibeiro7/genSurvPy/commit/e5114deb150e6a594d6b649049abd9ebd6242faa))

- Github address
  ([`43f9d93`](https://github.com/DiogoRibeiro7/genSurvPy/commit/43f9d93e21a285df6d5ea27d296ac489782b6b13))

### Chores

- Bump version
  ([`e0c739f`](https://github.com/DiogoRibeiro7/genSurvPy/commit/e0c739ff2c4aa8a4d46416aff5924969872d0337))

- Code coverage
  ([`56a3453`](https://github.com/DiogoRibeiro7/genSurvPy/commit/56a345302c50be21cc05a5afa0ac406224dd77a0))

- Update pyproject
  ([`a4b25e4`](https://github.com/DiogoRibeiro7/genSurvPy/commit/a4b25e470954091254b1384a44a991a47341bf80))

### Continuous Integration

- Add GitHub Actions workflow for test automation
  ([`0d57884`](https://github.com/DiogoRibeiro7/genSurvPy/commit/0d57884f84e3d8e09a130e1eb87895cd168ab1e0))

- Add GitHub Actions workflow for test automation
  ([`33e1e40`](https://github.com/DiogoRibeiro7/genSurvPy/commit/33e1e400ace7b491d88138331df496f7b7ab02c9))

### Documentation

- Add Sphinx configuration with Markdown support and index.md setup
  ([`75d4653`](https://github.com/DiogoRibeiro7/genSurvPy/commit/75d46530cf894263b3e98cf8667ff52c7863f646))

- Add Sphinx documentation with MyST and autodoc integration
  ([`f58feea`](https://github.com/DiogoRibeiro7/genSurvPy/commit/f58feeae82cb2b9073d936412a494a4a74ff9df1))

### Features

- Add changelog automation
  ([`055c209`](https://github.com/DiogoRibeiro7/genSurvPy/commit/055c20934b76c71fa1a91bf7948cd3699398a242))

- Add changelog automation
  ([`f22aba7`](https://github.com/DiogoRibeiro7/genSurvPy/commit/f22aba754b7fc118477f9070ae0d7a5377a9659d))

- Implement core CPHM data simulation with validation and censoring models
  ([`f5ef282`](https://github.com/DiogoRibeiro7/genSurvPy/commit/f5ef2829c2896c5ca575382b9b2d9e389784496d))

- Implement THMM data generator and finalize full model suite
  ([`1e667ba`](https://github.com/DiogoRibeiro7/genSurvPy/commit/1e667babf28892c3a85c43477562f2de85f07f3c))
