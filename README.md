# gen_surv

[![Coverage](https://codecov.io/gh/DiogoRibeiro7/genSurvPy/branch/main/graph/badge.svg)](https://app.codecov.io/gh/DiogoRibeiro7/genSurvPy)
[![Docs](https://readthedocs.org/projects/gensurvpy/badge/?version=latest)](https://gensurvpy.readthedocs.io/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/gen_surv)](https://pypi.org/project/gen-surv/)
[![Tests](https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/gen_surv)](https://pypi.org/project/gen-surv/)

**gen_surv** is a Python package for simulating survival data under a variety of statistical models. It is inspired by the R package [genSurv](https://cran.r-project.org/package=genSurv) and provides a unified interface for generating realistic survival datasets.

---

## Features

- Cox proportional hazards model (CPHM)
- Accelerated failure time models (log-normal, log-logistic)
- Continuous-time multi-state Markov model (CMM)
- Time-dependent covariate model (TDCM)
- Time-homogeneous hidden Markov model (THMM)
- Mixture cure and piecewise exponential models
- Competing risks generators (constant and Weibull hazards)
- Command-line interface and export utilities
- Scikit-learn compatible data generator
- Conversion helper for scikit-survival and lifelines

## Installation

Requires Python 3.10 or later.

Install the latest release from PyPI:

```bash
pip install gen-surv
```

To develop locally with all extras:

```bash
git clone https://github.com/DiogoRibeiro7/genSurvPy.git
cd genSurvPy
# Install runtime and development dependencies
# (scikit-survival is optional but required for integration tests).
# On Debian/Ubuntu you may need ``build-essential gfortran libopenblas-dev`` to
# build scikit-survival.
poetry install --with dev
```

Integration tests that rely on scikit-survival are automatically skipped if the
package is not installed.

## Development Setup

Before committing changes, install the pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Quick Example

```python
from gen_surv import export_dataset, generate

# basic Cox proportional hazards data
sim = generate(
    model="cphm",
    n=100,
    beta=0.5,
    covariate_range=2.0,
    model_cens="uniform",
    cens_par=1.0,
)

# save to an RDS file
export_dataset(sim, "survival_data.rds")
```

You can also convert the resulting DataFrame for use with
[scikit-survival](https://scikit-survival.readthedocs.io) or
[lifelines](https://lifelines.readthedocs.io):

```python
from gen_surv import to_sksurv

sks_dataset = to_sksurv(sim)
```

See the [usage guide](docs/source/getting_started.md) for more examples.

## Supported Models

| Model                | Description                             |
|----------------------|-----------------------------------------|
| **CPHM**             | Cox proportional hazards                |
| **AFT**              | Accelerated failure time (log-normal, log-logistic) |
| **CMM**              | Continuous-time multi-state Markov      |
| **TDCM**             | Time-dependent covariates                |
| **THMM**             | Time-homogeneous hidden Markov          |
| **Competing Risks**  | Multiple event types with cause-specific hazards |
| **Mixture Cure**     | Models long-term survivors              |
| **Piecewise Exponential** | Flexible baseline hazard via intervals |

More details on each algorithm are available in the [Algorithms](docs/source/algorithms.md) page and the [theory guide](docs/source/theory.md).

## Command-Line Usage

Datasets can be generated without writing Python code:

```bash
python -m gen_surv dataset cphm --n 1000 -o survival.csv
```

## Documentation

Full documentation is hosted on [Read the Docs](https://gensurvpy.readthedocs.io/en/latest/). It includes installation instructions, tutorials, API references and a bibliography.

To build the docs locally:

```bash
cd docs
make html
```

Open `build/html/index.html` in your browser to view the result.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use **gen_surv** in your research, please cite the project using the metadata in [CITATION.cff](CITATION.cff).

## Author

**Diogo Ribeiro** — [ESMAD - Instituto Politécnico do Porto](https://esmad.ipp.pt)

- ORCID: <https://orcid.org/0009-0001-2022-7072>
- Professional email: <dfr@esmad.ipp.pt>
- Personal email: <diogo.debastos.ribeiro@gmail.com>

