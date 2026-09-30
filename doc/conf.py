# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'OptiCut'
copyright = '2025, ONERA and MINES PARIS - PSL'
author = 'Amina El Bachari (ONERA & MINES Paris - PSL)'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

bibtex_bibfiles = ["reference.bib"]

extensions = [
    'nbsphinx',  # Support for Jupyter notebooks
    'sphinx.ext.mathjax',  # (Optional) Support for math formulas
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for NumPy/Google docstring styles
    'sphinx.ext.viewcode',  # Adds links to source code
    'sphinx.ext.autosectionlabel',  # Adds clickable references for figures
    'sphinxcontrib.bibtex', 
    'myst_parser',  # Support for Markdown
    'sphinxcontrib.mermaid',  # Support for Mermaid diagrams
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

autosectionlabel_prefix_document = True

autosectionlabel_enabled = False
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_logo = 'images/opticut-logo-v2.svg'
html_title = 'OptiCut'

# Optional Furo theme tweaks
html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#000000",
        "color-brand-content": "#000000",
    },
}
html_static_path = ['_static']

nbsphinx_allow_errors = True


import os

import sys

sys.path.insert(0, os.path.abspath('../src'))

# Add the static directory path
html_static_path = ['_static']

# Add custom CSS
html_css_files = [
    'custom.css',
]

numfig = True
numfig_format = {'figure': 'Figure %s'}

from docutils.parsers.rst import directives
html_css_files = ['custom.css']
