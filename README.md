![Woogle Maps Logo](https://raw.githubusercontent.com/TeamEpochGithub/woogle-maps/main/assets/Woogle_Maps_Logo_Auto.svg)

[![Epoch](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FJeffrey-Lim%2Fepoch-dvdscreensaver%2Fmaster%2Fbadge.json)](https://teamepoch.ai/)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TeamEpochGithub/woogle-maps/main.svg)](https://results.pre-commit.ci/latest/github/TeamEpochGithub/woogle-maps/main)
[![codecov](https://codecov.io/gh/TeamEpochGithub/woogle-maps/graph/badge.svg)](https://codecov.io/gh/TeamEpochGithub/woogle-maps)

Woogle Maps is  a tool for structuring documents in multiple clear storylines, visualizing the relationships between them.

This tool was made by Team Epoch IV for the [Rijksoverheid](https://www.rijksoverheid.nl/) and the [National Archive](https://www.nationaalarchief.nl/).
Read more about this [here](https://teamepoch.ai/competitions#Government).

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our results.

### Prerequisites

The project was developed using Python 3.12.3, Pip 24.0, and Poetry 1.7.1. Make sure all these are installed on your machine first.

Clone the repository with your favourite git client or using the following command:

```bash
git clone https://github.com/TeamEpochGithub/woogle-maps.git
```

### Installing dependencies

Run the following command to install all the dependencies:

```bash
poetry install
```

Alternatively, you can install the dependencies from `requirements.txt` using the following command:

```bash
pip install -r requirements.txt
```

Then activate the virtual environment:

```bash
poetry shell
```

### Running the Web App

To run the web app, simply run following command:

```bash
python dashapp/app.py
```

This will start the web app on `http://localhost:8060/`.

## Pytest coverage report

To generate pytest coverage report run:

```shell
poetry run pytest --cov=src --cov-branch --cov-report=html:coverage_re
```

## Documentation

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To make the documentation, run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in Google Chrome:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```
