# ✅ Python Package Development Checklist

A checklist to ensure quality, maintainability, and usability of your Python package.

---

## 1. Purpose & Scope

- [ ] Clear purpose and use cases defined  
- [ ] Scoped to a specific problem/domain  
- [ ] Project name is meaningful and not taken on PyPI  

---

## 2. Project Structure

- [ ] Uses `src/` layout or appropriate flat structure  
- [ ] All package folders contain `__init__.py`  
- [ ] Main configuration handled via `pyproject.toml`  
- [ ] Includes standard files: `README.md`, `LICENSE`, `.gitignore`, `CHANGELOG.md`  

---

## 3. Dependencies

- [ ] All dependencies declared in `pyproject.toml` or `requirements.txt`  
- [ ] Development dependencies separated from runtime dependencies  
- [ ] Uses minimal, necessary dependencies only  

---

## 4. Code Quality

- [ ] Follows PEP 8 formatting  
- [ ] Imports sorted with `isort` or `ruff`  
- [ ] No linter warnings (`ruff`, `flake8`, etc.)  
- [ ] Fully typed using `typing` module  
- [ ] No unresolved TODOs or FIXME comments  

---

## 5. Function & Module Design

- [ ] Functions are small, pure, and single-responsibility  
- [ ] Classes follow clear and simple roles  
- [ ] Global state is avoided  
- [ ] Public API defined explicitly (e.g. via `__all__`)  

---

## 6. Documentation

- [ ] `README.md` includes overview, install, usage, contributing  
- [ ] All functions/classes include docstrings (Google/NumPy style)  
- [ ] API reference documentation auto-generated (e.g., Sphinx, MkDocs)  
- [ ] Optional: `docs/` folder for additional documentation or site generator  

---

## 7. Testing

- [ ] Unit and integration tests implemented  
- [ ] Test coverage > 80% verified by `coverage`  
- [ ] Tests are fast and deterministic  
- [ ] Continuous Integration (CI) configured to run tests  

---

## 8. Versioning & Releases

- [ ] Uses semantic versioning (MAJOR.MINOR.PATCH)  
- [ ] Git tags created for releases  
- [ ] `CHANGELOG.md` updated with each release  
- [ ] Local build verified (`poetry build`, `hatch build`, or equivalent)  
- [ ] Can be published to PyPI and/or TestPyPI  

---

## 9. CLI or Scripts (Optional)

- [ ] CLI entrypoint works correctly (`__main__.py` or `entry_points`)  
- [ ] CLI provides helpful messages (`--help`) and handles errors gracefully  

---

## 10. Examples / Tutorials

- [ ] Usage examples included in `README.md` or `examples/`  
- [ ] Optional: Jupyter notebooks with demonstrations  
- [ ] Optional: Colab or Binder links for live usage  

---

## 11. Licensing & Attribution

- [ ] LICENSE file included (MIT, Apache 2.0, GPL, etc.)  
- [ ] Author and contributors credited in `README.md`  
- [ ] Optional: `CITATION.cff` file for academic citation  

---

> You can duplicate this file for each new package or use it as a GitHub issue template for release checklists.
