# ruff: noqa: INP001

"""Sphinx configuration file."""

import importlib
import inspect
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from pygit2 import Repository

REPO_URL: Final[str] = f"https://github.com/TeamEpochGithub/woogle-maps/-/blob/{Repository('.').head.shorthand}/"

sys.path.insert(0, Path("../..").resolve().as_posix())

project: Final[str] = "Woogle Maps"
copyright: Final[str] = "2024, Team Epoch"  # noqa: A001
author: Final[str] = "Team Epoch"

source_suffix: Final[dict[str, str]] = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc: Final[str] = "index"

extensions: Final[list[str]] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "myst_parser",
    "sphinxawesome_theme.highlighting",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.bibtex",
]
autosummary_generate: Final[bool] = True
autodoc_typehints: Final[str] = "signature"

autodoc_default_options: Final[dict[str, bool | str]] = {
    "members": True,
    "undoc-members": True,
    "member-order": "bysource",
}


def linkcode_resolve(domain: str, info: Mapping[str, str]) -> str | None:
    """Determine the URL corresponding to the Python object.

    This is used by sphinx.ext.linkcode to generate links to our source code.

    The code for getting the line numbers is copied from https://github.com/python-websockets/websockets/blob/main/docs/conf.py.

    :param domain: domain of the object
    :param info: information about the object
    :return: URL to the object or None if it is not Python code
    """
    if domain != "py":
        return None
    if not info["module"]:
        return None

    module = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        obj_name, attr_name = info["fullname"].split(".")
        obj: Any = getattr(module, obj_name)
        try:
            # Object is a method of a class
            obj = getattr(obj, attr_name)
        except AttributeError:
            # Object is an attribute of a class
            return None
    else:
        obj = getattr(module, info["fullname"])

    try:
        lines: tuple[list[str], int] = inspect.getsourcelines(obj)
    except TypeError:
        # E.g. object is a typing.Union
        return None

    start: int = lines[1]
    end: int = lines[1] + len(lines[0]) - 1

    filename: str = info["module"].replace(".", "/")
    return f"{REPO_URL}{filename}.py#L{start}-L{end}"


pygments_style: Final[str] = "sphinx"

templates_path: Final[list[str]] = ["_templates"]
exclude_patterns: Final[list[str]] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme: Final[str] = "sphinxawesome_theme"
html_theme_options: Final[dict[str, str]] = {"logo_light": "../assets/Woogle_Maps_Icon_Dark.svg", "logo_dark": "../assets/Woogle_Maps_Icon_Light.svg"}
html_css_files: Final[list[str]] = ["./logo.css"]
html_favicon: Final[str] = "../assets/Woogle_Maps_Icon_Auto.svg"
html_static_path: Final[list[str]] = ["_static", "../assets"]
html_use_smartypants: Final[bool] = True
html_show_sourcelink: Final[bool] = True
html_show_sphinx: Final[bool] = True
html_show_copyright: Final[bool] = True
bibtex_bibfiles = ["bibliography.bib"]
