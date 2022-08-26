# s2spy

A high-level python package integrating expert knowledge and artificial intelligence to boost (sub) seasonal forecasting.

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/AI4S2S/ai4s2s)
[![github license badge](https://img.shields.io/github/license/AI4S2S/s2spy)](https://github.com/AI4S2S/s2spy)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu)
[![build](https://github.com/AI4S2S/s2spy/actions/workflows/build.yml/badge.svg)](https://github.com/AI4S2S/s2spy/actions/workflows/build.yml)
[![sonarcloud](https://github.com/AI4S2S/s2spy/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/AI4S2S/s2spy/actions/workflows/sonarcloud.yml)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=AI4S2S_ai4s2s&metric=coverage)](https://sonarcloud.io/dashboard?id=AI4S2S_ai4s2s)

## Why s2spy?
Producing reliable (sub) seasonal (S2S) forecasts with machine learning techniques remains a huge challenge. Currently, these data-driven S2S forecasts generally suffers from a lack of trust because:
- Intransparent data processing and poorly reproducible scientific outcomes
- Technical pitfalls related to machine learning based predictability (e.g. overfitting)
- Black-box methods without sufficient explanation

To tackle these challenges, we build this open-source, high-level python package, `s2spy`. It provides an interface between artificial intelligence and expert knowledge, to boost predictability and physical understanding of S2S processes. By implementing optimal data-handling and parallel-computing packages, it can efficiently run across different Big Climate Data platforms. Key components will be explainable AI and causal discovery, which will support the classical scientific interplay between theory, hypothesis-generation and data-driven hypothesis-testing, enabling knowledge-mining from data.

This tool will be a community effort. It will help us achieve trustworthy data-driven forecasts by providing the user with:
- Transparent and reproducible analyses
- Best practices in model verification
- Understanding of the sources of predictability

## Installation
[![workflow pypi badge](https://img.shields.io/pypi/v/s2spy.svg?colorB=blue)](https://pypi.python.org/project/s2spy/)
[![supported python versions](https://img.shields.io/pypi/pyversions/dianna)](https://pypi.python.org/project/s2spy/)

To install the latest release of s2spy, do:
```console
python3 -m pip install s2spy
```

To install the in-development version from the GitHub repository, do:

```console
git clone https://github.com/AI4S2S/s2spy.git
cd s2spy
python3 -m pip install .
```

### Configure the package for development and testing
The testing framework used here is [pytest](https://pytest.org). Before running the test, the package need to be installed and configured as via the command:

```py
pip install -e .
```
or
```py
python setup.py develop
```

## Getting started
- work with xarray
- about calendar
- resample data
- cross-validation

Working progress, more to be added.

## Tutorials
TODO: add link to all tutorials and add link to api from readthedoc.

## Documentation
[![Documentation Status](https://readthedocs.org/projects/ai4s2s/badge/?version=latest)](https://ai4s2s.readthedocs.io/en/latest/?badge=latest)

The full documentation of the `s2spy` package can be found [here](https://ai4s2s.readthedocs.io/en/latest/).

## Contributing

If you want to contribute to the development of s2spy,
have a look at the [contribution guidelines](docs/CONTRIBUTING.md).

## How to cite us
<!-- [![RSD](https://img.shields.io/badge/rsd-s2s-00a3e3.svg)](https://www.research-software.nl/software/s2spy) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) -->
TODO: add links to zenodo and rsd.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
