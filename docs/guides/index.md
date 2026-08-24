# Guides

Task-oriented pages: everything that is not "which model should I use".

<div class="grid cards" markdown>

-   :material-clipboard-check: **[Configuration and ground truth](simulation-results.md)**

    `simulate()` returns the coefficients, latent times and cure status the
    frame cannot show — what a simulation study is actually about.

-   :material-chart-bell-curve: **[Baseline hazards](baselines.md)**

    The five hazard families every sampler draws from, and how to supply one of
    your own.

-   :material-scissors-cutting: **[Censoring](censoring.md)**

    The two built-in mechanisms, hitting a target event rate, and applying a
    censoring distribution of your own.

-   :material-view-column: **[Covariates](covariates.md)**

    Three different covariate schemes across the model families, and how to
    control each.

-   :material-table-search: **[Summarising a dataset](summaries.md)**

    Event counts, follow-up, quality checks, and comparing several datasets at
    once.

-   :material-chart-line: **[Plotting](plotting.md)**

    Kaplan-Meier curves, stratified curves, hazard comparisons, covariate
    effects.

-   :material-content-save: **[Exporting data](export.md)**

    CSV, JSON, Feather and RDS — and what a round trip loses.

-   :material-swap-horizontal: **[Fitting models to the data](interoperability.md)**

    lifelines, scikit-survival, scikit-learn and R.

-   :material-console: **[Command line](cli.md)**

    Generating and plotting without writing Python.

</div>

## Runnable notebooks

Three worked notebooks live in the repository under `examples/notebooks/`, and
open in a browser with no install through Binder:

| Notebook | Opens in Binder |
|---|---|
| Illness-death, intervals (`cmm`) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiogoRibeiro7/genSurvPy/HEAD?urlpath=lab/tree/examples/notebooks/cmm.ipynb) |
| Illness-death, panel (`thmm`) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiogoRibeiro7/genSurvPy/HEAD?urlpath=lab/tree/examples/notebooks/thmm.ipynb) |
| Time-dependent covariates (`tdcm`) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiogoRibeiro7/genSurvPy/HEAD?urlpath=lab/tree/examples/notebooks/tdcm.ipynb) |

The same directory also holds plain scripts — `run_cphm.py`, `run_aft.py`,
`run_competing_risks.py` and others — that can be run directly from a checkout.
