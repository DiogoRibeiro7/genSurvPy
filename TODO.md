# gen_surv Roadmap

This document outlines the planned development priorities for future versions of gen_surv. This roadmap will be periodically updated based on user feedback, research needs, and community contributions.

## Short-term Goals (v1.1.x)

### Additional Statistical Models
- [ ] **Recurrent Events Model**: Generate data with multiple events per subject
- [ ] **Time-Varying Effects**: Support for non-proportional hazards with coefficients that change over time
- [ ] **Extended Competing Risks**: Allow for correlation between competing risks

### Visualization and Analysis
- [ ] **Enhanced Visualization Toolkit**: Add more plot types and customization options
- [ ] **Interactive Visualizations**: Add options using Plotly for interactive exploration
- [ ] **Data Quality Reports**: Generate reports on statistical properties of generated datasets

### Usability Improvements
- [ ] **Dataset Catalog**: Pre-configured parameters to mimic classic survival datasets
- [ ] **Parameter Estimation**: Tools to estimate generation parameters from existing datasets
- [ ] **Extended CLI**: Add more command-line options for all models

## Medium-term Goals (v1.2.x)

### Advanced Statistical Models
- [ ] **Joint Longitudinal-Survival Models**: Generators for models that simultaneously handle longitudinal outcomes and time-to-event data
- [ ] **Frailty Models**: Support for shared and nested frailty models
- [ ] **Interval Censoring**: Support for interval-censored data generation

### Technical Enhancements
- [ ] **Parallel Processing**: Multi-core support for faster generation of large datasets
- [ ] **Memory Optimization**: Streaming data generation for very large datasets
- [ ] **Performance Benchmarks**: Systematic benchmarking of data generation speed

### Integration and Ecosystem
- [ ] **scikit-learn Extensions**: More scikit-learn compatible estimators and transformers
- [ ] **Stan/PyMC Integration**: Export data in formats suitable for Bayesian modeling
- [ ] **Dashboard**: Simple Streamlit app for data exploration and generation

## Long-term Goals (v2.x)

### Advanced Features
- [ ] **Bayesian Survival Models**: Generators for Bayesian survival analysis with various priors
- [ ] **Spatial Survival Models**: Generate survival data with spatial correlation
- [ ] **Survival Neural Networks**: Integration with deep learning approaches to survival analysis

### Infrastructure and Performance
- [ ] **GPU Acceleration**: Optional GPU support for large dataset generation
- [ ] **JAX/Numba Implementation**: High-performance implementations of key algorithms
- [ ] **R Interface**: Create an R package that interfaces with gen_surv

### Community and Documentation
- [ ] **Interactive Tutorials**: Using Jupyter Book or similar tools
- [ ] **Video Tutorials**: Short video demonstrations of key features
- [ ] **Case Studies**: Real-world examples showing how gen_surv can be used for teaching or research
- [ ] **User Showcase**: Gallery of research or teaching that uses gen_surv

## How to Contribute

We welcome contributions that help us achieve these roadmap goals! If you're interested in working on any of these features, please check the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines and open an issue to discuss your approach before submitting a pull request.

For suggesting new features or modifications to this roadmap, please open an issue with the "enhancement" tag.

## Version History

For a detailed history of past releases, please see our [CHANGELOG.md](CHANGELOG.md).
