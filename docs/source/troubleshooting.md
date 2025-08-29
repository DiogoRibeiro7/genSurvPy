# Troubleshooting

Common issues and how to resolve them when using gen_surv.

## ModuleNotFoundError: No module named 'gen_surv'

Ensure the package is installed. If you're running from source, install in editable mode:

```bash
pip install -e .
```

or with Poetry:

```bash
poetry install
```

## Validation errors when generating data

Many generators validate their inputs and raise ``ValidationError`` with
context such as ``while validating inputs for model 'cphm'``. Verify that
numeric parameters are within the allowed ranges and that sequences have the
correct length.

## Inconsistent results between runs

Most generators accept a ``seed`` parameter. Set it for reproducibility:

```python
from gen_surv import generate

df = generate(model="cphm", n=100, beta=0.5, covariate_range=2.0,
              model_cens="uniform", cens_par=1.0, seed=42)
```

