# gen_surv: Survival Data Simulation in Python

[![Documentation Status](https://readthedocs.org/projects/gensurvpy/badge/?version=latest)](https://gensurvpy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/gen-surv.svg)](https://badge.fury.io/py/gen-surv)
[![Python versions](https://img.shields.io/pypi/pyversions/gen-surv.svg)](https://pypi.org/project/gen-surv/)

**gen_surv** is a comprehensive Python package for simulating survival data under various statistical models, inspired by the R package `genSurv`. It provides a unified interface for generating synthetic survival datasets that are essential for:

- **Research**: Testing new survival analysis methods
- **Education**: Teaching survival analysis concepts
- **Benchmarking**: Comparing different survival models
- **Validation**: Testing statistical software implementations

```{admonition} Quick Start
:class: tip

Install with pip:
```bash
pip install gen-surv
```

Generate your first dataset:
```python
from gen_surv import generate
df = generate(model="cphm", n=100, beta=0.5, covariate_range=2.0)
```
```

## Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| **CPHM** | Cox Proportional Hazards | Standard survival regression |
| **AFT** | Accelerated Failure Time | Non-proportional hazards |
| **CMM** | Continuous-Time Markov | Multi-state processes |
| **TDCM** | Time-Dependent Covariates | Dynamic risk factors |
| **THMM** | Time-Homogeneous Markov | Hidden state processes |
| **Competing Risks** | Multiple event types | Cause-specific hazards |
| **Mixture Cure** | Long-term survivors | Logistic cure fraction |
| **Piecewise Exponential** | Piecewise constant hazard | Flexible baseline |

## Algorithm Descriptions

For a brief summary of each statistical model see {doc}`algorithms`. Mathematical
details and notation are provided on the {doc}`theory` page.

## Documentation Contents

```{toctree}
:maxdepth: 2

getting_started
tutorials/index
api/index
theory
algorithms
examples/index
rtd
contributing
changelog
bibliography
```

## Quick Examples

### Cox Proportional Hazards Model
```python
import gen_surv as gs

# Basic CPHM with uniform censoring
df = gs.generate(
    model="cphm", 
    n=500, 
    beta=0.5, 
    covariate_range=2.0,
    model_cens="uniform", 
    cens_par=3.0
)
```

### Accelerated Failure Time Model
```python
# AFT with log-normal distribution
df = gs.generate(
    model="aft_ln",
    n=200,
    beta=[0.5, -0.3, 0.2],
    sigma=1.0,
    model_cens="exponential",
    cens_par=2.0
)
```

### Multi-State Markov Model
```python
# Three-state illness-death model
df = gs.generate(
    model="cmm",
    n=300,
    qmat=[[0, 0.1], [0.05, 0]],
    p0=[1.0, 0.0],
    model_cens="uniform",
    cens_par=5.0
)
```

## Key Features

- **Unified Interface**: Single `generate()` function for all models
- **Flexible Censoring**: Support for uniform and exponential censoring
- **Rich Parameterization**: Extensive customization options
- **Command-Line Interface**: Generate datasets from terminal
- **Comprehensive Validation**: Input parameter checking
- **Educational Focus**: Clear mathematical documentation

## Citation

If you use gen_surv in your research, please cite:

```bibtex
@software{ribeiro2025gensurvpy,
  title = {gen_surv: Survival Data Simulation in Python},
  author = {Diogo Ribeiro},
  year = {2025},
  url = {https://github.com/DiogoRibeiro7/genSurvPy},
  version = {1.0.8}
}
```

## License

MIT License - see [LICENSE](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/LICENCE) for details.

For foundational papers related to these models see the {doc}`bibliography`.
Information on building the docs is provided in the {doc}`rtd` page.
