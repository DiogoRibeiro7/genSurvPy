---
orphan: true
---

# Read the Docs

This project uses [Read the Docs](https://readthedocs.org/) to host its documentation. The site is automatically rebuilt whenever changes are pushed to the repository.

Our build configuration is defined in `.readthedocs.yml`. It installs the package with the `docs` dependency group and builds the Sphinx docs using Python 3.11.

## Building Locally

To preview the documentation on your machine, run:

```bash
cd docs
make html
```

Open `build/html/index.html` in your browser to view the result.
