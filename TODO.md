# TODO – Roadmap for gen_surv

This document outlines future enhancements, features, and ideas for improving the gen_surv package.

---

## 📦 1. Interface and UX

- [ ] Create a `generate(..., return_type="df" | "dict")` interface
- [ ] Add `__version__` using `importlib.metadata` or `poetry-dynamic-versioning`
- [ ] Build a CLI with `typer` or `click`
- [ ] Add example notebooks for each model (`notebooks/` folder)

---

## 📚 2. Documentation

- [ ] Add a "Model Comparison Guide" section
- [ ] Add "How It Works" sections for each model
- [ ] Include usage tutorials in Jupyter format on RTD
- [ ] Optional: add multilingual docs using `sphinx-intl`

---

## 🧪 3. Testing and Quality

- [ ] Add property-based tests with `hypothesis`
- [ ] Cover edge cases (e.g., invalid parameters, n=0, negative censoring)
- [ ] Run tests on multiple Python versions (CI matrix)

---

## 🧠 4. Advanced Models

- [ ] Add Piecewise Exponential Model support
- [ ] Add competing risks / multi-event simulation
- [ ] Implement parametric AFT models (log-normal, log-logistic)
- [ ] Simulate time-varying hazards
- [ ] Add informative or covariate-dependent censoring

---

## 📊 5. Visualization and Analysis

- [ ] Create `plot_survival(df, model=...)` utilities
- [ ] Create `describe_survival(df)` summary helpers
- [ ] Export data to CSV / JSON / Feather

---

## 🌍 6. Ecosystem Integration

- [ ] Add a `GenSurvDataGenerator` compatible with `sklearn`
- [ ] Enable use with `lifelines`, `scikit-survival`, `sksurv`
- [ ] Export in R-compatible formats (.csv, .rds)

---

## 🔁 7. Other Ideas

- [ ] Add performance benchmarks for each model
- [ ] Improve PyPI discoverability (add keywords)
- [ ] Create a Streamlit or Gradio live demo

---

## 🧠 8. New Survival Models to Implement

- [ ] Accelerated Failure Time (AFT) models:
  - [ ] Log-Normal AFT
  - [ ] Log-Logistic AFT
  - [ ] Weibull AFT formulation
- [ ] Piecewise Exponential Model
- [ ] Competing Risks simulation
- [ ] Recurrent Events simulation
- [ ] Mixture Cure Model
