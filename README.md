# gen_surv

![Coverage](https://codecov.io/gh/DiogoRibeiro7/genSurvPy/branch/main/graph/badge.svg)
[![Docs](https://readthedocs.org/projects/gensurvpy/badge/?version=stable)](https://gensurvpy.readthedocs.io/en/stable/)

**gen_surv** is a Python package for simulating survival data under a variety of models, inspired by the R package [`genSurv`](https://cran.r-project.org/package=genSurv). It supports data generation for:

- Cox Proportional Hazards Models (CPHM)
- Continuous-Time Markov Models (CMM)
- Time-Dependent Covariate Models (TDCM)
- Time-Homogeneous Hidden Markov Models (THMM)

---

## 📦 Installation

```bash
poetry install
```
## ✨ Features

- Consistent interface across models  
- Censoring support (`uniform` or `exponential`)  
- Easy integration with `pandas` and `NumPy`  
- Suitable for benchmarking survival algorithms and teaching 

## 🧪 Example

```python
from gen_surv.cphm import gen_cphm

df = gen_cphm(
    n=100,
    model_cens="uniform",
    cens_par=1.0,
    beta=0.5,
    covar=2.0
)
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

## 🔧 Available Generators

| Function     | Description                                |
|--------------|--------------------------------------------|
| `gen_cphm()` | Cox Proportional Hazards Model             |
| `gen_cmm()`  | Continuous-Time Multi-State Markov Model   |
| `gen_tdcm()` | Time-Dependent Covariate Model             |
| `gen_thmm()` | Time-Homogeneous Markov Model              |


```text
genSurvPy/
├── gen_surv/             # Pacote principal
│   ├── __main__.py       # Interface CLI via python -m
│   ├── cphm.py
│   ├── cmm.py
│   ├── tdcm.py
│   ├── thmm.py
│   ├── censoring.py
│   ├── bivariate.py
│   ├── validate.py
│
├── tests/                # Testes automatizados
│   ├── test_cphm.py
│   ├── test_cmm.py
│   ├── test_tdcm.py
│   ├── test_thmm.py
│
├── examples/             # Exemplos de uso
│   ├── run_cphm.py
│   ├── ...
│
├── pyproject.toml        # Configurado com Poetry
├── README.md
├── LICENSE
├── .gitignore
```

## 🧠 License

MIT License. See [LICENSE](LICENSE) for details.
