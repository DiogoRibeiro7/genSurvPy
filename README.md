# gen_surv

[![Coverage][cov-badge]][cov-link]
[![Docs][docs-badge]][docs-link]
[![PyPI][pypi-badge]][pypi-link]
[![Tests][ci-badge]][ci-link]
[![Python][py-badge]][pypi-link]

[cov-badge]: https://codecov.io/gh/DiogoRibeiro7/genSurvPy/branch/main/graph/badge.svg
[cov-link]: https://app.codecov.io/gh/DiogoRibeiro7/genSurvPy
[docs-badge]: https://readthedocs.org/projects/gensurvpy/badge/?version=latest
[docs-link]: https://gensurvpy.readthedocs.io/en/latest/
[pypi-badge]: https://img.shields.io/pypi/v/gen_surv
[pypi-link]: https://pypi.org/project/gen-surv/
[ci-badge]: https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml/badge.svg
[ci-link]: https://github.com/DiogoRibeiro7/genSurvPy/actions/workflows/ci.yml
[py-badge]: https://img.shields.io/pypi/pyversions/gen_surv
[![GitHub stars](https://img.shields.io/github/stars/diogoribeiro7/genSurvPy.svg?style=social)](https://github.com/diogoribeiro7/genSurvPy/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/diogoribeiro7/genSurvPy.svg?style=social)](https://github.com/diogoribeiro7/genSurvPy/network/members)

**gen_surv** is a Python library for simulating survival data and producing visualizations under a wide range of statistical models. Inspired by the R package [genSurv](https://cran.r-project.org/package=genSurv), it offers a unified interface for generating realistic datasets for research, teaching and benchmarking.

---

## Features

- Cox proportional hazards model (CPHM)
- Accelerated failure time models (log-normal, log-logistic, Weibull)
- Continuous-time multi-state Markov model (CMM)
- Time-dependent covariate model (TDCM)
- Time-homogeneous hidden Markov model (THMM)
- Mixture cure and piecewise exponential models
- Competing risks generators (constant and Weibull hazards)
- Visualization helpers built on matplotlib and lifelines
- Scikit-learn compatible data generator
- Conversion utilities for scikit-survival
- Command-line interface for dataset creation and visualization

## Installation

Requires Python 3.10 or later.

Install the latest release from PyPI:

```bash
pip install gen-surv
```

`gen_surv` installs matplotlib and lifelines for visualization. Support for scikit-survival is optional; install it to enable integration with the scikit-survival ecosystem or to run the full test suite:

```bash
pip install gen-surv[dev]
```

To develop locally with all extras:

```bash
git clone https://github.com/DiogoRibeiro7/genSurvPy.git
cd genSurvPy
poetry install --with dev
```

On Debian/Ubuntu you may need `build-essential gfortran libopenblas-dev` to build scikit-survival.

## Development

Before committing changes, install the pre-commit hooks and run the tests:

```bash
pre-commit install
pre-commit run --all-files
pytest
```

Tests that depend on optional packages such as scikit-survival are skipped automatically when those packages are missing.

## Usage

### Python API

```python
from gen_surv import generate, export_dataset, to_sksurv
from gen_surv.visualization import plot_survival_curve

sim = generate(
    model="cphm",
    n=100,
    beta=0.5,
    covariate_range=2.0,
    model_cens="uniform",
    cens_par=1.0,
)

plot_survival_curve(sim)
export_dataset(sim, "survival_data.rds")

# convert for scikit-survival
sks_dataset = to_sksurv(sim)
```

See the [usage guide](https://gensurvpy.readthedocs.io/en/latest/getting_started.html) for more examples.

### Command Line

Generate datasets and plots without writing Python code:

```bash
python -m gen_surv dataset cphm --n 1000 -o survival.csv

python -m gen_surv visualize survival.csv --output survival_plot.png
```

`visualize` accepts custom column names via `--time-col` and `--status-col` and can stratify by group with `--group-col`.

## Supported Models

| Model | Description |
|-------|-------------|
| **CPHM** | Cox proportional hazards |
| **AFT** | Accelerated failure time (log-normal, log-logistic, Weibull) |
| **CMM** | Continuous-time multi-state Markov |
| **TDCM** | Time-dependent covariates |
| **THMM** | Time-homogeneous hidden Markov |
| **Competing Risks** | Multiple event types with cause-specific hazards |
| **Mixture Cure** | Models long-term survivors |
| **Piecewise Exponential** | Flexible baseline hazard via intervals |

More details on each algorithm are available in the [Algorithms](https://gensurvpy.readthedocs.io/en/latest/algorithms.html) page. For additional background, see the [theory guide](https://gensurvpy.readthedocs.io/en/latest/theory.html).

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
- GitHub: [@DiogoRibeiro7](https://github.com/DiogoRibeiro7)
