# Benchmarks

Performance benchmarks for **genSurvPy** powered by the [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) plugin.

## Running

Install the optional `pytest-benchmark` dependency and execute:

```bash
poetry run pytest benchmarks -q --benchmark-only
```

To run an individual benchmark module:

```bash
poetry run pytest benchmarks/test_cmm_benchmark.py --benchmark-only
```

## Available benchmarks

- Validation helpers (`test_validation_benchmark.py`)
- Time-dependent Cox model generation (`test_tdcm_benchmark.py`)
- Continuous-time Markov model generation (`test_cmm_benchmark.py`)
- Piecewise exponential generation (`test_piecewise_benchmark.py`)
- Cox proportional hazards model generation (`test_cphm_benchmark.py`)
- Mixture cure model generation (`test_mixture_benchmark.py`)
