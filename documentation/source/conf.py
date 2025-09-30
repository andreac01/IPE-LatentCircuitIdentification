# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import sys
# The path to your project's root directory, one level up from the 'source' directory.
sys.path.insert(0, os.path.abspath('../../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IPE'
copyright = '2025, Andrea Cerutti. Released under the GNU General Public License v3.0'
author = 'Andrea Cerutti'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',       # Core library for autodoc
	'sphinx.ext.autosummary',   # Create summary tables
	'sphinx.ext.napoleon',      # Support for Google and NumPy style docstrings
	'sphinx.ext.viewcode',      # Add links to highlighted source code
	'sphinx.ext.intersphinx',   # Link to other projects' documentation
	'sphinx_autodoc_typehints', # Automatically document typehints
	'sphinx_copybutton',        # Add a "copy" button to code blocks
	'sphinx_design',            # For UI components like cards and grids
	'nbsphinx',              	# To include Jupyter Notebooks
]

templates_path = ['_templates']
exclude_patterns = []

# -- Autodoc & Autosummary settings ------------------------------------------
autodoc_default_options = {
	'members': True,
	'undoc-members': True,
	'private-members': False,
	'special-members': '__init__',
	'show-inheritance': True,
}
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# -- Napoleon settings for Google/NumPy-style docstrings ---------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# -- Intersphinx mapping -----------------------------------------------------
# Link to other projects' documentation
intersphinx_mapping = {
	'python': ('https://docs.python.org/3', None),
	'numpy': ('https://numpy.org/doc/stable/', None),
	'scipy': ('https://docs.scipy.org/doc/scipy/', None),
	'matplotlib': ('https://matplotlib.org/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Set custom logo and favicon paths
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

# Theme-specific options for PyData theme
html_theme_options = {
	"logo": {
		"text": "IPE Documentation",
	},
	"icon_links": [
		{
			"name": "GitHub",
			"url": "https://github.com/andreac01/IPE-LatentCircuitIdentification.git", 
			"icon": "fab fa-github-square",
			"type": "fontawesome",
		}
	],
	"navbar_align": "left",
	"navbar_end": ["theme-switcher", "navbar-icon-links"],
	"secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
	"show_toc_level": 2,
	"use_edit_page_button": True,
}


# Required for "edit-this-page" link
html_context = {
	"github_user": "acerutti",
	"github_repo": "https://github.com/andreac01/IPE-LatentCircuitIdentification.git",
	"github_version": "main",
	"doc_path": "source",
}

# -- Syntax Highlighting settings --------------------------------------------
# Use a style with good contrast for both light and dark modes.
pygments_style = 'friendly'
pygments_dark_style = 'monokai'
highlight_language = 'python3'


# -- Copybutton settings -----------------------------------------------------
# Exclude prompts and output when copying code.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
