# Installation

## From PyPI

```bash
pip install gen-surv
```

That is everything you need. The generators, the plotting helpers and the
command line tool all work from this single install.

!!! info "Python support"

    `gen_surv` requires **Python 3.11, 3.12 or 3.13**. It is tested on all
    three on every commit.

## What comes with it

These are hard dependencies, installed for you:

| Package | Used for |
|---|---|
| `numpy` | random number generation and the sampling routines |
| `pandas` | every generator returns a `DataFrame` |
| `scipy` | distribution functions used by several models |
| `matplotlib` | the plotting helpers in `gen_surv.visualization` |
| `lifelines` | Kaplan-Meier and hazard estimation behind those plots |
| `typer` / `click` | the `gen_surv` command line tool |
| `pyarrow` | Feather export |
| `pyreadr` | RDS export, for handing data to R |

## The one optional extra

[scikit-survival](https://scikit-survival.readthedocs.io) is **not** installed
by default. You need it only for the two conversion helpers,
[`to_sksurv`](../api/interoperability.md#gen_surv.integration.to_sksurv) and
[`from_sksurv`](../api/interoperability.md#gen_surv.integration.from_sksurv):

```bash
pip install scikit-survival
```

Without it, `import gen_surv` still succeeds — the package detects the missing
dependency and simply does not expose those two names.

!!! tip "scikit-survival needs a compiler on some platforms"

    It builds against C extensions. If `pip` struggles, conda-forge ships
    prebuilt wheels: `conda install -c conda-forge scikit-survival`.

## From source

```bash
git clone https://github.com/DiogoRibeiro7/genSurvPy.git
cd genSurvPy
poetry install
```

Add the groups you need:

```bash
poetry install --with dev    # pytest, mypy, black, isort, flake8, hypothesis, scikit-survival
poetry install --with docs   # mkdocs-material and mkdocstrings, to build this site
```

## Verifying the install

```python
import gen_surv

print(gen_surv.__version__)
print(len(gen_surv.__all__), "public names")
```

Or from the shell:

```bash
gen_surv --help
gen_surv dataset cphm --n 5
```

## Building the documentation

This site is built with [MkDocs](https://www.mkdocs.org) and
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/):

```bash
poetry install --with docs
poetry run mkdocs serve      # live-reloading preview on http://127.0.0.1:8000
poetry run mkdocs build      # static site into ./site
```

The published site is rebuilt automatically whenever a release is published, so
it always documents the version that is on PyPI.
