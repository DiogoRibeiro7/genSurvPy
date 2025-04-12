# gen_surv

**gen_surv** is a Python package for simulating survival data under various models, inspired by the R package `genSurv`.

It includes generators for:

- **Cox Proportional Hazards Models (CPHM)**
- **Continuous-Time Markov Models (CMM)**
- **Time-Dependent Covariate Models (TDCM)**
- **Time-Homogeneous Hidden Markov Models (THMM)**

---

## 📚 Modules

```{toctree}
:maxdepth: 2
:caption: Contents

modules
theory
```


# 🚀 Usage Example

```python
from gen_surv.cphm import gen_cphm

df = gen_cphm(n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)
print(df.head())
```

```python
from gen_surv import generate

df = generate(
    model="cphm",
    n=100,
    model_cens="uniform",
    cens_par=1.0,
    beta=0.5,
    covar=2.0
)

print(df.head())
```

## 🔗 Project Links

- [Source Code](https://github.com/DiogoRibeiro7/genSurvPy)
- [License](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/LICENSE)
