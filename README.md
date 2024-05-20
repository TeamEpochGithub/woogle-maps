![Woogle Maps Logo](https://raw.githubusercontent.com/TeamEpochGithub/woogle-maps/main/assets/Woogle_Maps_Logo_Auto.svg)

[![Epoch](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FJeffrey-Lim%2Fepoch-dvdscreensaver%2Fmaster%2Fbadge.json)](https://teamepoch.ai/)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/TeamEpochGithub/woogle-maps/main.svg)](https://results.pre-commit.ci/latest/github/TeamEpochGithub/woogle-maps/main)
[![codecov](https://codecov.io/gh/TeamEpochGithub/woogle-maps/graph/badge.svg?token=VCIQ3UDFUI)](https://codecov.io/gh/TeamEpochGithub/woogle-maps)

Woogle Maps is a tool for structuring documents in multiple clear storylines, visualizing the relationships between them.

On the basis of a list of PDF documents as input, Woogle Maps automatically generates a TimeFlow as output.
This TimeFlow – a timeline with a graphical structure – describes the events that underlie these documents,
and attaches every document to at least one event.
Woogle Maps encompasses an extension of the original Narrative Maps algorithm created by Brian Keith Norambuena.

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our results.

### Prerequisites

The project was developed using Python 3.12.3, Pip 24.0, and Poetry 1.8.2. Make sure all these are installed on your machine first.

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

The documentation is hosted on [GitHub Pages](https://teamepochgithub.github.io/woogle-maps/).

Documentation is generated using [Sphinx](https://www.sphinx-doc.org/en/master/).

To rebuild it, make sure to install the dependencies for generating the documentation first:

```shell
poetry install --with docs
```

Run `make html` with `docs` as the working directory. The documentation can then be found in `docs/_build/html/index.html`.

Here's a short command to make the documentation and open it in Google Chrome:

```shell
cd ./docs/;
./make.bat html; start chrome file://$PWD/_build/html/index.html
cd ../
```

## Contributors

Woogle Maps was created by [Team Epoch IV](https://teamepoch.ai/team), based in the [Dream Hall](https://www.tudelft.nl/ddream) of the [Delft University of Technology](https://www.tudelft.nl/).
This project was commisioned by [Nationaal Archief](https://www.nationaalarchief.nl/) and financed by [i-Partnerschap](https://www.rijksorganisatieodi.nl/i-partnerschap).

Read more about this [here](https://teamepoch.ai/competitions#Government).

- Team Lead: [Jeffrey Lim](https://www.linkedin.com/in/jeffrey-si-hau-lim/) (Epoch IV)
- Team Members: [Daniel de Dios Allegue](https://www.linkedin.com/in/daniel-de-dios-allegue/) (Epoch IV), [Kristóf Sándor](https://www.linkedin.com/in/kristof-sandor/) (Epoch IV), and [Gregoire Dumont](https://www.linkedin.com/in/gregoire-dumont-592586240/) (Epoch IV)
- Supervisor: [Max Muller](https://www.linkedin.com/in/max-muller-2861625b/) (Nationaal Archief)
- Advisors: [Erik Saaman](https://www.linkedin.com/in/erik-saaman-5246624/) (Nationaal Archief) and [Paul van den Akker](https://www.linkedin.com/in/paulvdakker/) (Nationaal Archief)
- Relationship managers: [Evelien Christiaanse](https://www.linkedin.com/in/evelienchristiaanse/) (i-Partnerschap), [Arlette Wegman](https://www.linkedin.com/in/arlettewegman/t) (i-Partnerschap), [Robert van Poeteren](https://www.linkedin.com/in/robert-van-poeteren-994041283/) (Epoch IV), [Brian Witmer](https://www.linkedin.com/in/brian-witmer-222028190/) (Epoch IV), and [Emiel Witting](https://www.linkedin.com/in/emiel-witting-3b515a290/) (Epoch IV)
