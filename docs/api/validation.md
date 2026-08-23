# Validation

Every generator validates its arguments before drawing anything, so an invalid
call fails immediately rather than returning quietly wrong data.

All errors derive from `ValidationError`, which derives from `ValueError` — so
`except ValueError` catches them all, and `except ValidationError` catches only
this package's.

```python
from gen_surv import generate
from gen_surv.validation import ValidationError

try:
    generate(model="cphm", n=-1, beta=0.5, covariate_range=2.0,
             model_cens="uniform", cens_par=1.0)
except ValidationError as exc:
    print(exc)
```

::: gen_surv.validation
