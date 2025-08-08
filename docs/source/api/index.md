---
orphan: true
---

# API Reference

Complete documentation for all gen_surv functions and classes.

```{note}
The `to_sksurv` helper and related tests rely on the optional
dependency `scikit-survival`. Install it with `poetry install --with dev`
or `pip install scikit-survival` to enable this functionality.
```

## Core Interface

::: gen_surv.interface
    options:
      members: true
      undoc-members: true
      show-inheritance: true

## Model Generators

### Cox Proportional Hazards Model
::: gen_surv.cphm
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Accelerated Failure Time Models
::: gen_surv.aft
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Continuous-Time Markov Models
::: gen_surv.cmm
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Time-Dependent Covariate Models
::: gen_surv.tdcm
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Time-Homogeneous Markov Models
::: gen_surv.thmm
    options:
      members: true
      undoc-members: true
      show-inheritance: true

## Utility Functions

### Censoring Functions
::: gen_surv.censoring
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Bivariate Distributions
::: gen_surv.bivariate
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Validation Functions
::: gen_surv.validation
    options:
      members: true
      undoc-members: true
      show-inheritance: true

### Command Line Interface
::: gen_surv.cli
    options:
      members: true
      undoc-members: true
      show-inheritance: true

