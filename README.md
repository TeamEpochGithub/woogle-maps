![Woogle Maps Logo](./assets/Woogle_Maps_Logo_Auto.svg)

[![Epoch](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2FJeffrey-Lim%2Fepoch-dvdscreensaver%2Fmaster%2Fbadge.json)](https://teamepoch.ai/)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Woogle Maps is a tool for structuring documents and visualizing the relationships between them.

## Getting started

This section contains the steps that need to be taken to get started with our project and fully reproduce our results.

### Prerequisites

The project was developed using Python 3.12.3, Pip 24.0, and Poetry 1.7.1. Make sure all these are installed on your machine first.

Clone the repository with your favourite git client or using the following command:

```bash
# TODO(Jeffrey): Update the git clone link to GitHub
git clone https://gitlab.ewi.tudelft.nl/dreamteam-epoch/epoch-iv/q3-national-archive.git
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
