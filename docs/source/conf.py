import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").absolute()))

project = "coastpy"
copyright = "2022, Floris Calkoen"
author = "Floris Calkoen"

extensions = [
    "myst_parser",  # For Markdown and Jupyter Notebook support
    "autoapi.extension",  # For API documentation
    "sphinx.ext.napoleon",  # For Google style docstrings
    "sphinx.ext.viewcode",  # To include the source code in the docs
    "nbsphinx",  # To include Jupyter Notebooks
    "sphinx_markdown_tables",  # To handle Markdown tables
]

autoapi_dirs = ["../../src"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

nbsphinx_allow_errors = True  # Continue through Jupyter errors

# Additional myst-parser configurations
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "html_image",
]
