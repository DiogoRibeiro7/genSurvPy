# Time-Dependent Covariate Model (TDCM)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiogoRibeiro7/genSurvPy/HEAD?urlpath=lab/tree/examples/notebooks/tdcm.ipynb)

A basic visualization of event times produced by the TDCM generator:

```python
import numpy as np
import matplotlib.pyplot as plt
from gen_surv import generate

np.random.seed(0)

df = generate(
    model="tdcm",
    n=200,
    dist="weibull",
    corr=0.5,
    dist_par=[1, 2, 1, 2],
    model_cens="uniform",
    cens_par=1.0,
    beta=[0.1, 0.2, 0.3],
    lam=1.0,
)

plt.hist(df["stop"], bins=20, color="#4C72B0")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("TDCM Event Times")
```
