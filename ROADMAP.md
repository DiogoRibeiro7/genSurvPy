# gen_surv Roadmap

This is the current project roadmap. `TODO.md` remains the long-form decision
log and historical backlog; this file is the shorter entry point for what comes
next.

For released work, see
[CHANGELOG.md](https://github.com/DiogoRibeiro7/genSurvPy/blob/develop/CHANGELOG.md).

## Current State

- `gen_surv` is a production/stable survival-data simulation package.
- Distribution tests, property-based tests and R parity fixtures cover the
  supported generators.
- The `discrimination-not-calibration` research study has completed its compact
  production run and generated manuscript-facing result artifacts.
- The production lock and raw/processed research outputs are local artifacts by
  design; generated paper tables and hypothesis macros are tracked.

## Near Term

- Prepare the next package release once the research/reporting changes are
  reviewed for release notes and versioning.
- Draft the `discrimination-not-calibration` manuscript prose from the generated
  tables, figures and hypothesis macros. Do not type numeric results by hand.
- Decide whether the `discrimination-not-calibration` paper should be added to
  the external `article-reminders` tracker, with its repository path,
  manuscript path and next action.
- Keep dependency maintenance moving through small, verified Dependabot PRs.

## Research Manuscript

- Replace placeholder manuscript sections with prose grounded in the committed
  protocol and generated artifacts.
- Keep the title focused on the measurement contribution rather than claiming
  that discrimination and calibration can disagree.
- Preserve the provenance boundary: production rows were generated under the
  compact experiment lock, and later table-formatting changes must not be
  described as changes to the simulation experiment.
- Before submission, verify the manuscript builds from a clean checkout plus the
  documented local research artifacts.

## Package Work

- Redesign the CLI around per-model subcommands so each simulator exposes only
  its own parameters.
- Add scenario-file support for reproducible, shareable simulation
  configurations.
- Move optional plotting, I/O and estimator-integration dependencies behind
  extras.
- Continue vectorising the remaining subject-level generator loops, using the
  benchmark suite to justify each change.

## Model Expansion

- Add frailty and clustered survival mechanisms.
- Add advanced censoring support: informative censoring, interval censoring,
  left truncation and dependent censoring.
- Add time-varying effects such as delayed effects, crossing hazards and change
  points.
- Add correlated competing risks through copulas or shared frailty.
- Add missingness and measurement-error mechanisms for benchmark datasets that
  resemble real studies.

## Research Framework

- Generalise the research study machinery into a reusable simulation-study
  runner.
- Add fit-to-simulation adapters for scenarios derived from `lifelines` or
  `scikit-survival` fitted models.
- Add exports shaped for Stan and PyMC workflows.

## Deliberately Not Planned

- GPU acceleration.
- Survival neural networks.
- Interactive dashboards or Plotly visualisations in the core package.
- An R interface.
- Video tutorials or a user showcase as core development priorities.

The reasons for these exclusions are recorded in
[TODO.md](https://github.com/DiogoRibeiro7/genSurvPy/blob/develop/TODO.md).
