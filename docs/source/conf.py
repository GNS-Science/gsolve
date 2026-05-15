# GSolve - gravity processing software.
# Copyright (c) 2026 Earth Sciences New Zealand.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPLv3

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "gSolve"
copyright = "2026, Adrian Benson, Alison Kirkby, Craig Miller, Aleksandr Spesivtsev, Vaughan Stagpoole, Earth Sciences New Zealand"
author = "Adrian Benson, Alison Kirkby, Craig Miller, Aleksandr Spesivtsev, Vaughan Stagpoole"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
from gsolve import __version__

release = __version__
# The short X.Y version.
# version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
]
numpydoc_show_class_members = False

myst_enable_extensions = ["amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_title = "gSolve"
html_title = "gSolve"
html_logo = "_static/gsolve_logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "show_navbar_depth": 1,
    "repository_provider": "github",
    "repository_url": "https://github.com/GNS-Science/gsolve",
    "repository_branch": "main",
    "use_repository_button": True,
    # Render version in the logo block (left sidebar) under the logo image.
    "logo": {
        "text": f"gSolve\nVersion {release}",
    },
}
html_context = {
    "display_version": True,  # Ensure version is displayed
    "version": release,  # Pass version variable to templates
    "version": release,  # Pass version variable to templates
}
