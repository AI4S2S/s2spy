# s2spy

A high-level python package integrating expert knowledge and artificial intelligence to boost (sub) seasonal forecasting.
About how to use the `s2spy` package, check the [documentation](https://ai4s2s.readthedocs.io/en/latest/index.html).

[![github repo badge](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue)](https://github.com/AI4S2S/ai4s2s)
[![workflow pypi badge](https://img.shields.io/pypi/v/s2spy.svg?colorB=blue)](https://pypi.python.org/project/s2spy/)
[![Documentation Status](https://readthedocs.org/projects/ai4s2s/badge/?version=latest)](https://ai4s2s.readthedocs.io/en/latest/?badge=latest)
[![github license badge](https://img.shields.io/github/license/AI4S2S/s2spy)](https://github.com/AI4S2S/s2spy)
[![fair-software badge](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow)](https://fair-software.eu) \
[![build](https://github.com/AI4S2S/s2spy/actions/workflows/build.yml/badge.svg)](https://github.com/AI4S2S/s2spy/actions/workflows/build.yml)
[![sonarcloud](https://github.com/AI4S2S/s2spy/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/AI4S2S/s2spy/actions/workflows/sonarcloud.yml)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=AI4S2S_ai4s2s&metric=coverage)](https://sonarcloud.io/dashboard?id=AI4S2S_ai4s2s)
<!-- [![RSD](https://img.shields.io/badge/rsd-s2s-00a3e3.svg)](https://www.research-software.nl/software/s2s) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/<replace-with-created-DOI>.svg)](https://doi.org/<replace-with-created-DOI>) -->

## Installation

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

## Configure the package for development and testing
The testing framework used here is [pytest](https://pytest.org). Before running the test, the package need to be installed and configured as via the command:

```py
pip install -e .
```
or
```py
python setup.py develop
```

## Documentation

The full documentation of the `s2spy` package can be found [here](https://ai4s2s.readthedocs.io/en/latest/).

## Contributing

If you want to contribute to the development of s2s,
have a look at the [contribution guidelines](docs/CONTRIBUTING.md).

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
