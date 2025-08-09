# Continuous-Time Multi-State Markov Model (CMM)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/DiogoRibeiro7/genSurvPy/HEAD?urlpath=lab/tree/examples/notebooks/cmm.ipynb)

Visualize transition times from the CMM generator:

```python
import numpy as np
import matplotlib.pyplot as plt
from gen_surv import generate

np.random.seed(0)

df = generate(
    model="cmm",
    n=200,
    model_cens="exponential",
    cens_par=2.0,
    beta=[0.1, 0.2, 0.3],
    covariate_range=1.0,
    rate=[0.1, 1.0, 0.2, 1.0, 0.1, 1.0],
)

plt.hist(df["stop"], bins=20, color="#4C72B0")
plt.xlabel("Time")
plt.ylabel("Frequency")
plt.title("CMM Transition Times")
```
