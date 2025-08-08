# Getting Started

This page offers a quick introduction to installing and using **gen_surv**.

## Installation

The project is managed with [Poetry](https://python-poetry.org). Clone the repository and install dependencies:

```bash
poetry install
```

This will create a virtual environment and install all required packages.

## Basic Usage

Generate datasets directly in Python:

```python
from gen_surv import generate

# Cox Proportional Hazards example
generate(model="cphm", n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)
```

You can also generate data from the command line:

```bash
python -m gen_surv dataset aft_ln --n 100 > data.csv
```

For a full description of available models and parameters, see the API reference.

