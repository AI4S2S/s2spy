# `s2spy` developer documentation

If you're looking for user documentation, go [here](readme_link.rst).

## Development install

```shell
# Create a virtual environment, e.g. with
python3 -m venv env

# activate virtual environment
source env/bin/activate

# make sure to have a recent version of pip and hatch
python3 -m pip install --upgrade pip hatch

# (from the project root directory)
# install s2spy as an editable package
python3 -m pip install --no-cache-dir --editable .
# install development dependencies
python3 -m pip install --no-cache-dir --editable .[dev]
```

Afterwards check that the install directory is present in the `PATH` environment variable.

## Running the tests

Running tests has been configured using `hatch`, and can be started by running:

```shell
hatch run test
```

The second is to use `tox`, which can be installed separately (e.g. with `pip install tox`), i.e. not necessarily inside the virtual environment you use for installing `s2spy`, but then builds the necessary virtual environments itself by simply running:

### Test coverage

In addition to just running the tests to see if they pass, they can be used for coverage statistics, i.e. to determine how much of the package's code is actually executed during tests.
Inside the package directory, run:

```shell
hatch run coverage
```

This runs tests and prints the results to the command line, as well as storing the result in a `coverage.xml` file (for analysis by, e.g. SonarCloud).

## Running linters locally

For linting we will use `flake8`, `black` and `isort`. We additionally use `mypy` to check the type hints.
All tools can simply be run by doing:

# linter
```shell
hatch run lint
```

To easily comply with `black` and `isort`, you can also run:

```shell
hatch run format
```

This will apply the `black` and `isort` formatting, and then check the code style.


## Generating the documentation
To generate the documentation, simply run the following command. This will also test the documentation code snippets. Note that you might need to install [`pandoc`](https://pandoc.org/) to be able to generate the documentation.

```shell
hatch run docs:build
```

The documentation will be in `docs/_build/html`.

## Versioning

Bumping the version across all files is done with [bumpversion](https://github.com/c4urself/bump2version), e.g.

```shell
bumpversion major
bumpversion minor
bumpversion patch
```

## Making a release

This section describes how to make a release in 3 parts:

1. preparation
1. making a release on PyPI
1. making a release on GitHub

### (1/3) Preparation

1. Update the <CHANGELOG.md> (don't forget to update links at bottom of page)
2. Verify that the information in `CITATION.cff` is correct, and that `.zenodo.json` contains equivalent data
3. Make sure the [version has been updated](#versioning).
4. Run the unit tests with `hatch run test`

### (2/3) PyPI

First prepare a new directory, for example:

```shell
# prepare a new directory
cd $(mktemp -d s2spy.XXXXXX)
```

A fresh git clone ensures the release has the state of origin/main branch

```shell
git clone https://github.com/AI4S2S/s2spy .
```

In a your terminal, with an activated environment which has [`hatch`](https://hatch.pypa.io/latest/) installed do:

```shell
pip install hatch --upgrade
hatch build
```

If the build was succesfull, publish it to [PyPI's test servers](https://test.pypi.org/). Note that your credentials are different between test.pypi.org and pypy.org.
```shell
hatch publish --repo test
```

Visit
[https://test.pypi.org/project/s2spy](https://test.pypi.org/project/s2spy)
and verify that your package was uploaded successfully. Keep the terminal open, we'll need it later.

Now we can publish to PyPI:
```shell
hatch publish
```

### (3/3) GitHub

Don't forget to also make a [release on GitHub](https://github.com/AI4S2S/s2spy/releases/new). If your repository uses the GitHub-Zenodo integration this will also trigger Zenodo into making a snapshot of your repository and sticking a DOI on it.
