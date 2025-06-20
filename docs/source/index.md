# gen_surv

**gen_surv** is a Python package for simulating survival data under various models, inspired by the R package `genSurv`.

It includes generators for:

- **Cox Proportional Hazards Models (CPHM)**
- **Continuous-Time Markov Models (CMM)**
- **Time-Dependent Covariate Models (TDCM)**
- **Time-Homogeneous Hidden Markov Models (THMM)**
- **Accelerated Failure Time (AFT) Log-Normal Models**

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
from gen_surv import generate

# CPHM
generate(model="cphm", n=100, model_cens="uniform", cens_par=1.0, beta=0.5, covar=2.0)

# AFT Log-Normal
generate(model="aft_ln", n=100, beta=[0.5, -0.3], sigma=1.0, model_cens="exponential", cens_par=3.0)

# CMM
generate(model="cmm", n=100, model_cens="exponential", cens_par=2.0,
         qmat=[[0, 0.1], [0.05, 0]], p0=[1.0, 0.0])

# TDCM
generate(model="tdcm", n=100, dist="weibull", corr=0.5,
         dist_par=[1, 2, 1, 2], model_cens="uniform", cens_par=1.0,
         beta=[0.1, 0.2, 0.3], lam=1.0)

# THMM
generate(model="thmm", n=100, qmat=[[0, 0.2, 0], [0.1, 0, 0.1], [0, 0.3, 0]],
         emission_pars={"mu": [0.0, 1.0, 2.0], "sigma": [0.5, 0.5, 0.5]},
         p0=[1.0, 0.0, 0.0], model_cens="exponential", cens_par=3.0)
```

## ⌨️ Command-Line Usage

Generate datasets directly from the terminal:

```bash
python -m gen_surv dataset aft_ln --n 100 > data.csv
```

## Repository Layout

```text
genSurvPy/
├── gen_surv/
│   └── ...
├── tests/
├── examples/
├── docs/
├── scripts/
├── tasks.py
└── TODO.md
```

## 🔗 Project Links

- [Source Code](https://github.com/DiogoRibeiro7/genSurvPy)
- [License](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/LICENCE)
- [Code of Conduct](https://github.com/DiogoRibeiro7/genSurvPy/blob/main/CODE_OF_CONDUCT.md)

## Citation

If you use **gen_surv** in your work, please cite it using the metadata in
[CITATION.cff](../../CITATION.cff).

## Author

**Diogo Ribeiro** — [ESMAD - Instituto Politécnico do Porto](https://esmad.ipp.pt)

- ORCID: <https://orcid.org/0009-0001-2022-7072>
- Professional email: <dfr@esmad.ipp.pt>
- Personal email: <diogo.debastos.ribeiro@gmail.com>
