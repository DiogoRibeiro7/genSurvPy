# gen_surv

**Simulate survival data with a known truth.**

[![PyPI][pypi-badge]][pypi-link]
[![Python][py-badge]][pypi-link]
[![Tests][ci-badge]][ci-link]
[![Coverage][cov-badge]][cov-link]
[![Docs][pages-badge]][pages-link]
[![License][license-badge]](LICENSE)

[pypi-badge]: https://img.shields.io/pypi/v/gen_surv
[pypi-link]: https://pypi.org/project/gen-surv/
[py-badge]: https://img.shields.io/pypi/pyversions/gen_surv
[ci-badge]: https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml
[cov-badge]: https://codecov.io/gh/DiogoRibeiro7/genSurvPy/branch/main/graph/badge.svg
[cov-link]: https://app.codecov.io/gh/DiogoRibeiro7/genSurvPy
[pages-badge]: https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/gh-pages.yml/badge.svg
[pages-link]: https://diogoribeiro7.github.io/genSurvPy/
[license-badge]: https://img.shields.io/pypi/l/gen_surv

`gen_surv` generates synthetic time-to-event datasets from twelve models —
proportional hazards, accelerated failure time, competing risks, cure
fractions, piecewise hazards, recurrent events and two illness-death
processes — so you can test an estimator against parameters you chose
yourself.

It is a Python port of the R package
[genSurv](https://cran.r-project.org/package=genSurv), extended well past the
original's four models.

📖 **[Documentation](https://diogoribeiro7.github.io/genSurvPy/)** ·
🚀 **[Quickstart](https://diogoribeiro7.github.io/genSurvPy/getting-started/quickstart/)** ·
🧪 **[Choosing a model](https://diogoribeiro7.github.io/genSurvPy/models/)**

## Install

```bash
pip install gen-surv
```

Python 3.11, 3.12 and 3.13. Everything is included except
[scikit-survival](https://scikit-survival.readthedocs.io), which is optional and
needed only for the two conversion helpers:

```bash
pip install scikit-survival
```

## Thirty seconds

```python
from gen_surv import generate

df = generate(
    model="cphm",           # Cox proportional hazards
    n=6,
    beta=0.5,               # log hazard ratio
    covariate_range=2.0,    # X0 ~ Uniform(0, 2)
    model_cens="uniform",
    cens_par=1.0,
    seed=42,
)
print(df)
```

```text
       time  status        X0
0  0.438878     0.0  1.547912
1  0.094177     0.0  1.394736
2  0.037041     1.0  1.522279
3  0.370798     0.0  0.900772
4  0.646901     1.0  1.287730
5  0.251113     1.0  0.454477
```

You picked `beta = 0.5`, so you know what a correct estimator should recover:

```python
from lifelines import CoxPHFitter

df = generate(model="cphm", n=5000, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=7)

CoxPHFitter().fit(df, duration_col="time", event_col="status").params_
# X0    0.501
```

That is the whole idea. Every column was produced by a mechanism you specified,
so anything an estimator gets wrong is the estimator's fault.

## The twelve models

| `model=` | Family | Rows per subject |
| --- | --- | --- |
| `cphm` | Cox proportional hazards | 1 |
| `aft_ln` | Log-normal AFT | 1 |
| `aft_weibull` | Weibull AFT | 1 |
| `aft_log_logistic` | Log-logistic AFT | 1 |
| `piecewise_exponential` | Piecewise constant hazard | 1 |
| `competing_risks` | Cause-specific constant hazards | 1 |
| `competing_risks_weibull` | Cause-specific Weibull hazards | 1 |
| `mixture_cure` | Logistic cure + exponential failure | 1 |
| `cmm` | Illness-death, counting-process intervals | 2 or 3 |
| `thmm` | Illness-death, observed state panel | 2 or 3 |
| `tdcm` | Cox with a time-dependent covariate | 1 |
| `recurrent_events` | Repeated events: Andersen-Gill, PWP | 1 per at-risk interval |

Every model has a page covering its parameters, the mathematics, a worked
example, and a check that the parameters can be recovered from the data it
generates — start at
[Choosing a model](https://diogoribeiro7.github.io/genSurvPy/models/).

> **The output shape is not the same for every model.** Multi-state generators
> return several rows per subject, and column names differ between families.
> See [Output schemas](https://diogoribeiro7.github.io/genSurvPy/getting-started/schemas/)
> before writing code that consumes a generated frame.

## The ground truth, not just the data

A generated frame looks like a real one — which means it hides the same things.
`simulate()` hands back what a real dataset never could:

```python
from gen_surv import simulate

result = simulate("cphm", n=1000, beta=0.5, covariate_range=2.0,
                  model_cens="uniform", cens_par=1.0, seed=42)

result.data                      # the frame generate() would return
result.config                    # model, parameters, seed, gen_surv version
result.truth["event_time"]       # when each subject would have failed
result.truth["censoring_time"]   # what censoring hid
```

Several models **draw their coefficients for you** when you leave them out.
`result.truth["betas"]` is the only way to learn what they were — without it
those datasets cannot validate anything.

## Any hazard shape

Every generator that draws a time inverts a cumulative hazard, so the shape is
a parameter rather than a fork in the code:

```python
from gen_surv import generate, LogLogisticBaseline

recurrent = generate(model="recurrent_events", n=500,
                     baseline=LogLogisticBaseline(shape=2.0, scale=1.5),
                     betas=[0.4, -0.2], followup_time=6.0, seed=1)
```

Exponential, Weibull, Gompertz, log-logistic and piecewise-constant are
built in, and anything implementing `hazard`, `cumulative_hazard` and its
inverse works too.

## Beyond generating

```python
from gen_surv import describe_survival, plot_survival_curve, export_dataset, to_sksurv

describe_survival(df)              # events, censoring, median follow-up
plot_survival_curve(df)            # Kaplan-Meier, optionally stratified
export_dataset(df, "data.rds")     # csv, json, feather or rds
to_sksurv(df)                      # structured array for scikit-survival
```

- **[Ground truth](https://diogoribeiro7.github.io/genSurvPy/guides/simulation-results/)** — configurations, latent times, the coefficients actually used
- **[Baseline hazards](https://diogoribeiro7.github.io/genSurvPy/guides/baselines/)** — the five families, and writing your own
- **[Censoring](https://diogoribeiro7.github.io/genSurvPy/guides/censoring/)** — the built-in mechanisms, hitting a target event rate, applying your own distribution
- **[Covariates](https://diogoribeiro7.github.io/genSurvPy/guides/covariates/)** — the three schemes across model families
- **[Summaries](https://diogoribeiro7.github.io/genSurvPy/guides/summaries/)** — event counts, quality checks, dataset comparison
- **[Plotting](https://diogoribeiro7.github.io/genSurvPy/guides/plotting/)** — survival curves, hazard comparisons, covariate effects
- **[Fitting models](https://diogoribeiro7.github.io/genSurvPy/guides/interoperability/)** — lifelines, scikit-survival, scikit-learn, R

## Command line

```bash
gen_surv dataset cphm --n 1000 --beta 0.5 --seed 42 -o survival.csv
gen_surv visualize survival.csv --group-col X0 --output km.png
```

Repeat a flag for list arguments — `--beta 0.5 --beta -0.3`. Every one of the
twelve models is reachable from the command line. Full option reference in the
[CLI guide](https://diogoribeiro7.github.io/genSurvPy/guides/cli/).

## Reproducibility

Every generator takes a `seed`, accepting an `int` or a
`numpy.random.Generator`. The same seed on the same version always gives the
same frame, on any platform.

A bug fix in a sampler changes the draws a seed produces, so **pin the version
alongside the seed** for anything that must reproduce:

```text
gen-surv==2.1.0
```

See
[Reproducibility](https://diogoribeiro7.github.io/genSurvPy/getting-started/reproducibility/).

## Documentation

**<https://diogoribeiro7.github.io/genSurvPy/>**

| Section | Contents |
| --- | --- |
| [Getting started](https://diogoribeiro7.github.io/genSurvPy/getting-started/) | Installation, quickstart, output schemas, reproducibility |
| [Models](https://diogoribeiro7.github.io/genSurvPy/models/) | Per-model parameters, mathematics, examples, recovery checks |
| [Guides](https://diogoribeiro7.github.io/genSurvPy/guides/) | Censoring, covariates, summaries, plotting, export, interoperability, CLI |
| [Theory](https://diogoribeiro7.github.io/genSurvPy/theory/) | The mathematics behind every generator, plus the bibliography |
| [API](https://diogoribeiro7.github.io/genSurvPy/api/) | Full reference, generated from the source |

Built with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
and [mkdocstrings](https://mkdocstrings.github.io), and rebuilt from the release
tag by the [Pages workflow](.github/workflows/gh-pages.yml) — so it documents
the version on PyPI, not unreleased work.

## Development

```bash
git clone https://github.com/DiogoRibeiro7/genSurvPy.git
cd genSurvPy
poetry install --with dev

pre-commit install
pre-commit run --all-files     # black, isort, flake8, mypy
pytest                         # tests needing optional packages skip themselves
```

Docs:

```bash
poetry install --with docs
poetry run mkdocs serve        # live reload on http://127.0.0.1:8000
```

On Debian and Ubuntu, building scikit-survival may need
`build-essential gfortran libopenblas-dev`.

Work happens on `develop`; `main` carries releases. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{ribeiro_gensurv,
  title   = {gen_surv: Survival Data Simulation in Python},
  author  = {Diogo Ribeiro},
  url     = {https://github.com/DiogoRibeiro7/genSurvPy},
  version = {2.1.0}
}
```

Machine-readable metadata: [CITATION.cff](CITATION.cff) and
[.zenodo.json](.zenodo.json).

## License

MIT — see [LICENSE](LICENSE).

## Author

**Diogo Ribeiro** — [ESMAD, Instituto Politécnico do Porto](https://esmad.ipp.pt)

- ORCID: <https://orcid.org/0009-0001-2022-7072>
- Email: <dfr@esmad.ipp.pt> · <diogo.debastos.ribeiro@gmail.com>
- GitHub: [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)

[![GitHub stars](https://img.shields.io/github/stars/diogoribeiro7/genSurvPy.svg?style=social)](https://github.com/diogoribeiro7/genSurvPy/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/diogoribeiro7/genSurvPy.svg?style=social)](https://github.com/diogoribeiro7/genSurvPy/network/members)
