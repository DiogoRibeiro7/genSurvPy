# TODO – Roadmap for gen_surv

This document outlines future enhancements, features, and ideas for improving the gen_surv package.

---

## 📦 1. Interface and UX

- [✅] Create a `generate(..., return_type="df" | "dict")` interface
- [ ] Add `__version__` using `importlib.metadata` or `poetry-dynamic-versioning`
- [ ] Build a CLI with `typer` or `click`
- [✅] Add example notebooks or scripts for each model (`examples/` folder)

---

## 📚 2. Documentation

- [✅] Add a "Model Comparison Guide" section (`index.md` + `theory.md`)
- [✅] Add "How It Works" sections for each model (`theory.md`)
- [✅] Include usage examples in index with real calls
- [ ] Optional: add multilingual docs using `sphinx-intl`

---

## 🧪 3. Testing and Quality

- [✅] Add tests for each model (e.g., `test_tdcm.py`, `test_thmm.py`, `test_aft.py`)
- [ ] Add property-based tests with `hypothesis`
- [ ] Cover edge cases (e.g., invalid parameters, n=0, negative censoring)
- [ ] Run tests on multiple Python versions (CI matrix)

---

## 🧠 4. Advanced Models

- [ ] Add Piecewise Exponential Model support
- [ ] Add competing risks / multi-event simulation
- [✅] Implement parametric AFT models (log-normal)
- [ ] Implement parametric AFT models (log-logistic, weibull)
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
- [✅] Improve PyPI discoverability (added tags, keywords, docs)
- [ ] Create a Streamlit or Gradio live demo

---

## 🧠 8. New Survival Models to Implement

- [✅] Log-Normal AFT
- [ ] Log-Logistic AFT
- [ ] Weibull AFT
- [ ] Piecewise Exponential
- [ ] Competing Risks
- [ ] Recurrent Events
- [ ] Mixture Cure Model

---

## 🧬 9. Advanced Data Simulation Features

- [ ] Recurrent events (multiple events per individual)
- [ ] Frailty models (random effects)
- [ ] Time-varying hazard functions
- [ ] Multi-line start-stop formatted data
- [ ] Competing risks with cause-specific hazards
- [ ] Simulate violations of PH assumption
- [ ] Grouped / clustered data generation
- [ ] Mixed covariates: categorical, continuous, binary
- [ ] Joint models (longitudinal + survival outcome)
- [ ] Controlled scenarios for robustness tests
