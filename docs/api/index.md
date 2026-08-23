# API reference

Generated from the source, so it is always in step with the version you have
installed. For task-oriented material, start from [Guides](../guides/index.md)
or the [model pages](../models/index.md).

## Everything in `gen_surv`

`from gen_surv import ...` gives you:

| Name | Kind | Page |
|---|---|---|
| `generate` | dispatcher over all eleven models | [generate](generate.md) |
| `gen_cphm` | Cox proportional hazards | [Generators](generators.md#gen_surv.cphm) |
| `gen_aft_log_normal`, `gen_aft_weibull`, `gen_aft_log_logistic` | AFT models | [Generators](generators.md#gen_surv.aft) |
| `gen_piecewise_exponential` | piecewise constant hazard | [Generators](generators.md#gen_surv.piecewise) |
| `gen_competing_risks`, `gen_competing_risks_weibull` | competing risks | [Generators](generators.md#gen_surv.competing_risks) |
| `gen_mixture_cure`, `cure_fraction_estimate` | cure models | [Generators](generators.md#gen_surv.mixture) |
| `gen_cmm` | illness-death, intervals | [Generators](generators.md#gen_surv.cmm) |
| `gen_thmm` | illness-death, panel | [Generators](generators.md#gen_surv.thmm) |
| `gen_tdcm` | time-dependent covariates | [Generators](generators.md#gen_surv.tdcm) |
| `runifcens`, `rexpocens`, `rweibcens`, `rlognormcens`, `rgammacens` | censoring samplers | [Censoring](censoring.md) |
| `WeibullCensoring`, `LogNormalCensoring`, `GammaCensoring`, `CensoringModel` | class-based censoring | [Censoring](censoring.md) |
| `sample_bivariate_distribution` | correlated draws | [Censoring](censoring.md#gen_surv.bivariate) |
| `describe_survival`, `plot_survival_curve`, `plot_hazard_comparison`, `plot_covariate_effect` | summaries and plots | [Analysis](analysis.md#gen_surv.visualization) |
| `export_dataset` | write to disk | [Interoperability](interoperability.md#gen_surv.export) |
| `to_sksurv`, `from_sksurv` | scikit-survival conversion — needs the optional dependency | [Interoperability](interoperability.md#gen_surv.integration) |
| `GenSurvDataGenerator` | scikit-learn estimator | [Interoperability](interoperability.md#gen_surv.sklearn_adapter) |
| `__version__` | the installed version | — |

## Not exported at the top level

These live one level down and are imported from their module:

| Import | Purpose |
|---|---|
| `from gen_surv.summary import summarize_survival_dataset, check_survival_data_quality, compare_survival_datasets` | dataset summaries and quality checks — [Analysis](analysis.md#gen_surv.summary) |
| `from gen_surv.validation import ValidationError, ...` | the exception hierarchy — [Validation](validation.md) |
| `from gen_surv.cli import app` | the Typer application — [Command line](cli.md) |

## Conventions across the package

**Every generator returns a `DataFrame`.** Never a numpy array, never a tuple.
Shapes differ by model — see [Output schemas](../getting-started/schemas.md).

**`seed` is always last and always optional.** It accepts an `int`, a
`numpy.random.Generator`, or `None`. See
[Reproducibility](../getting-started/reproducibility.md).

**Validation happens before any sampling.** An invalid argument raises a
subclass of `ValidationError` — itself a `ValueError` — with the offending
argument, the value received, and what was expected.

**Optional dependencies degrade quietly.** `to_sksurv` and `from_sksurv` are
simply absent when scikit-survival is not installed; importing `gen_surv` still
works.
