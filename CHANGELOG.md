# CHANGELOG

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
