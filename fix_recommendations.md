# Fixing gen_surv Repository Issues

## Priority 1: Critical Fixes

### 1. Fix `__init__.py` Import Issues

Update `gen_surv/__init__.py` to include missing imports:

```python
# Add these imports
from .aft import gen_aft_log_logistic
from .competing_risks import gen_competing_risks, gen_competing_risks_weibull
from .mixture import gen_mixture_cure, cure_fraction_estimate
from .piecewise import gen_piecewise_exponential

# Update __all__ to include:
__all__ = [
    # ... existing exports ...
    "gen_aft_log_logistic",
    "gen_competing_risks", 
    "gen_competing_risks_weibull",
    "gen_mixture_cure",
    "cure_fraction_estimate", 
    "gen_piecewise_exponential",
]
```

### 2. Add Missing Validators

Create validation functions in `validate.py`:

```python
def validate_gen_aft_weibull_inputs(n, beta, shape, scale, model_cens, cens_par):
    # Add validation logic
    
def validate_gen_aft_log_logistic_inputs(n, beta, shape, scale, model_cens, cens_par):
    # Add validation logic
    
def validate_competing_risks_inputs(n, n_risks, baseline_hazards, betas, model_cens, cens_par):
    # Add validation logic
```

### 3. Update CLI Integration

Modify `cli.py` to handle all available models:

```python
# Add support for competing_risks, mixture, and piecewise models
# Update parameter handling for each model type
```

## Priority 2: Version Consistency

### Update Version Numbers

1. **CITATION.cff**: Change version from "1.0.3" to "1.0.8"
2. **docs/source/conf.py**: Change release from '1.0.3' to '1.0.8'

## Priority 3: Testing and Documentation

### Add Missing Tests

Create test files:
- `tests/test_censoring.py`
- `tests/test_mixture.py` 
- `tests/test_piecewise.py`
- `tests/test_summary.py`
- `tests/test_visualization.py`

### Update Documentation

1. Add competing risks, mixture, and piecewise models to `theory.md`
2. Update examples in documentation to include new models

## Priority 4: Code Quality Improvements

### Standardize Parameter Naming

- Consistently use `covariate_cols` instead of mixing `covar` and `covariate_cols`
- Standardize return column names across all models

### Add Type Hints

Complete type hints for all public functions, especially in:
- `mixture.py`
- `piecewise.py` 
- `summary.py`

## Verification Steps

After implementing fixes:

1. **Test imports**: `python -c "from gen_surv import gen_aft_log_logistic, gen_competing_risks"`
2. **Test CLI**: `python -m gen_surv dataset competing_risks --n 10`
3. **Run full test suite**: `poetry run pytest`
4. **Check version consistency**: `python scripts/check_version_match.py`
5. **Build docs**: `poetry run sphinx-build docs/source docs/build`

## Impact Assessment

These fixes will:
- ✅ Eliminate ImportError exceptions for users
- ✅ Make all models accessible via public API
- ✅ Ensure CLI works for all supported models
- ✅ Improve user experience and API consistency
- ✅ Maintain backward compatibility