# Contributing to gen_surv

Thank you for taking the time to contribute to **gen_surv**! This document provides a brief overview of the recommended workflow for feature requests and pull requests.

## Getting Started

1. Fork the repository and create your feature branch from `main`.
2. Install dependencies with `poetry install --with dev`.
   This installs all packages needed for development, including
   the optional dependency `scikit-survival`.
   On Debian/Ubuntu you may need `build-essential gfortran libopenblas-dev`
   to build it.
3. Run `pre-commit install` to enable style checks and execute them with `pre-commit run --all-files`.
4. Ensure the test suite passes with `poetry run pytest`.
5. If you add a feature or fix a bug, update `CHANGELOG.md` accordingly.

## Submitting Changes

1. Commit your changes with clear messages.
2. Push to your branch and open a pull request.
3. Ensure your PR description explains the motivation and summarizes your changes.

We appreciate your contributions and feedback!
