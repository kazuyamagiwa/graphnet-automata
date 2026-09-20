# Configuration file for the Sphinx documentation builder.
#
# Builds a Read the Docs–style site from Markdown (MyST) sources and
# autodoc-generated API pages for the graphnet_automata package.

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make src/ importable when building docs without an editable install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "graphnet-automata"
author = "Masakazu Yamagiwa"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"
version = "0.1"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "graphnet-automata documentation"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# Avoid executing heavy plotting imports during autodoc where possible.
autodoc_mock_imports = []
