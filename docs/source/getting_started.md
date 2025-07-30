---
orphan: true
---

# Getting Started

This guide will help you install gen_surv and generate your first survival dataset.

## Installation

### From PyPI (Recommended)

```bash
pip install gen-surv
```

### From Source

```bash
git clone https://github.com/DiogoRibeiro7/genSurvPy.git
cd genSurvPy
poetry install
```

## Basic Usage

The main entry point is the `generate()` function:

```python
from gen_surv import generate

# Generate Cox proportional hazards data
 df = generate(
     model="cphm",      # Model type
     n=100,             # Sample size
     beta=0.5,          # Covariate effect
     covar=2.0,         # Covariate range
     model_cens="uniform",  # Censoring type
     cens_par=3.0       # Censoring parameter
 )

print(df.head())
```

## Understanding the Output

All models return a pandas DataFrame with at least these columns:

- `time`: Observed event or censoring time
- `status`: Event indicator (1 = event, 0 = censored)
- Additional columns depend on the specific model

## Command Line Usage

Generate datasets directly from the terminal:

```bash
# Generate CPHM data and save to CSV
python -m gen_surv dataset cphm --n 1000 -o survival_data.csv

# Print AFT data to stdout
python -m gen_surv dataset aft_ln --n 500
```

## Next Steps

- Explore the {doc}`tutorials/index` for detailed examples
- Check the {doc}`api/index` for complete function documentation
- Read about the {doc}`theory` behind each model

## Building the Documentation

To preview the documentation locally run:

```bash
cd docs
make html
```

More details about our Read the Docs configuration can be found in {doc}`rtd`.

