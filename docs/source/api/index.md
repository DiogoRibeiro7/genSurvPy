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

```{eval-rst}
.. automodule:: gen_surv.interface
   :members:
   :undoc-members:
   :show-inheritance:
```

## Model Generators

### Cox Proportional Hazards Model
```{eval-rst}
.. automodule:: gen_surv.cphm
   :members:
   :undoc-members:
   :show-inheritance:
```

### Accelerated Failure Time Models
```{eval-rst}
.. automodule:: gen_surv.aft
   :members:
   :undoc-members:
   :show-inheritance:
```

### Continuous-Time Markov Models
```{eval-rst}
.. automodule:: gen_surv.cmm
   :members:
   :undoc-members:
   :show-inheritance:
```

### Time-Dependent Covariate Models
```{eval-rst}
.. automodule:: gen_surv.tdcm
   :members:
   :undoc-members:
   :show-inheritance:
```

### Time-Homogeneous Markov Models
```{eval-rst}
.. automodule:: gen_surv.thmm
   :members:
   :undoc-members:
   :show-inheritance:
```

## Utility Functions

### Censoring Functions
```{eval-rst}
.. automodule:: gen_surv.censoring
   :members:
   :undoc-members:
   :show-inheritance:
```

### Bivariate Distributions
```{eval-rst}
.. automodule:: gen_surv.bivariate
   :members:
   :undoc-members:
   :show-inheritance:
```

### Validation Functions
```{eval-rst}
.. automodule:: gen_surv.validate
   :members:
   :undoc-members:
   :show-inheritance:
```

### Command Line Interface
```{eval-rst}
.. automodule:: gen_surv.cli
   :members:
   :undoc-members:
   :show-inheritance:
```
