# Fixing gen_surv Repository Issues

## Priority 1: Critical Fixes

- [x] **Fix `__init__.py` Import Issues**
  - Ensure missing imports for new generators are added and exported via `__all__`.

- [x] **Add Missing Validators**
  - Create validation helpers for AFT Weibull, AFT log-logistic, and competing risks generators.

- [x] **Update CLI Integration**
  - Support competing risks, mixture cure, and piecewise exponential models.

## Priority 2: Version Consistency

- [x] **Update Version Numbers**
  - `CITATION.cff` and `docs/source/conf.py` now reference version 1.0.8.

## Priority 3: Testing and Documentation

- [x] **Add Missing Tests**
  - Added tests for censoring helpers, mixture cure, piecewise exponential, summary, and visualization.

- [x] **Update Documentation**
  - Documented competing risks, mixture cure, and piecewise exponential models.

## Priority 4: Code Quality Improvements

- [x] **Standardize Parameter Naming**
  - Replaced the `covar` parameter with `covariate_range` and standardized return columns to `X0`.

- [x] **Add Type Hints**
  - Completed type hints for public functions in `mixture.py`, `piecewise.py`, and `summary.py`.

## Verification Steps
- [x] `python -c "from gen_surv import gen_aft_log_logistic, gen_competing_risks"`
- [x] `python -m gen_surv dataset competing_risks --n 10`
- [x] `pytest -q`
- [x] `python scripts/check_version_match.py`
- [x] `sphinx-build docs/source docs/build`

## Status

All fix recommendations have been implemented in version 1.0.8.

Verified as of commit `7daa3e1`.
