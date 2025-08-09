---
orphan: true
---

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
from gen_surv import export_dataset, generate

# Cox Proportional Hazards example
df = generate(
    model="cphm",
    n=100,
    model_cens="uniform",
    cens_par=1.0,
    beta=0.5,
    covariate_range=2.0,
)

# Save to RDS for use in R
export_dataset(df, "simulated_data.rds")
```

You can also generate data from the command line:

```bash
python -m gen_surv dataset aft_ln --n 100 > data.csv
```

For a full description of available models and parameters, see the API reference.


## Building the Documentation

Documentation is written using [Sphinx](https://www.sphinx-doc.org). To build the HTML pages locally run:

```bash
cd docs
make html
```

The generated files will be available under `docs/build/html`.

## Scikit-learn Integration

You can wrap the generator in a transformer compatible with scikit-learn:

```python
from gen_surv import GenSurvDataGenerator

est = GenSurvDataGenerator("cphm", n=10, beta=0.5, covariate_range=1.0)
df = est.fit_transform()
```

## Lifelines and scikit-survival

Datasets generated with **gen_surv** can be directly used with
[lifelines](https://lifelines.readthedocs.io). For
[scikit-survival](https://scikit-survival.readthedocs.io) you can convert the
DataFrame using ``to_sksurv``:

```{note}
The ``to_sksurv`` helper requires the optional dependency
``scikit-survival``. Install it with `poetry install --with dev` or
``pip install scikit-survival``.
```

```python
from gen_surv import to_sksurv

struct = to_sksurv(df)
```

