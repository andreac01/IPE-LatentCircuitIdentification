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
sys.path.insert(0, os.path.abspath('../../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'IPE'
copyright = '2025, Andrea Cerutti. Released under the GNU General Public License v3.0'
author = 'Andrea Cerutti'
release = '0.0.1'  # Use a more specific release number

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode', # Adds links to source code
    'sphinx.ext.intersphinx', # Links to other projects' docs
    'sphinx_copybutton', # Adds a "copy" button to code blocks
    'sphinx_design', # Adds components like cards and grids
]

templates_path = ['_templates']
exclude_patterns = []

# -- Napoleon settings for Google/NumPy-style docstrings ---------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = False # Use False if you prefer Google-style
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True # Renders examples in a nice admonition block

# -- Intersphinx mapping -----------------------------------------------------
# Link to other projects' documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Set custom logo and favicon paths
# You must have these files in your 'source/_static/' directory
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.ico"

# Theme-specific options for PyData theme
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/your-username/your-repo",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        }
    ],
    "navbar_align": "left",
    "secondary_sidebar_items": ["page-toc"],
    "footer_items": ["copyright", "last-updated"],
}

# Set the syntax highlighting style for light and dark modes
pygments_style = 'tango'
pygments_dark_style = 'native'
highlight_language = 'python'

# -- Copybutton settings -----------------------------------------------------
# Skip the dollar sign and other prompts when copying code
copybutton_prompt_text = "$ "