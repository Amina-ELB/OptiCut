# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'OptiCut'
copyright = '2025, ONERA and MINES PARIS - PSL'
author = 'ONERA and MINES PARIS - PSL'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

bibtex_bibfiles = ["reference.bib"]

extensions = [
    'nbsphinx',  # Support pour les notebooks Jupyter
    'sphinx.ext.mathjax',  # (Optionnel) Pour le support des formules
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Pour le support NumPy/Google style
    'sphinx.ext.viewcode', # Ajoute un lien vers le code source
    'sphinx.ext.autosectionlabel',  #to add clickable reference for figures
    'sphinxcontrib.bibtex', 
]

autosectionlabel_prefix_document = True

autosectionlabel_enabled = False
nbsphinx_execute = 'never'

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

nbsphinx_allow_errors = True


import os
import sphinx_rtd_theme  # Si vous utilisez le thème ReadTheDocs

# Ajouter le chemin du dossier _static
html_static_path = ['_static']

# Ajouter le CSS personnalisé
html_css_files = [
    'custom.css',
]

numfig = True
numfig_format = {'figure': 'Figure %s'}

from docutils.parsers.rst import directives
