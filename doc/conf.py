# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import datetime

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------

# The full version, including alpha/beta/rc tags.
import h5netcdf

release = h5netcdf.__version__

project = "h5netcdf"
copyright = "2015-%s, h5netcdf developers" % datetime.datetime.now().year
language = "en"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.githubpages",
]

extlinks = {
    "issue": ("https://github.com/h5netcdf/h5netcdf/issues/%s", "GH%s"),
    "pull": ("https://github.com/h5netcdf/h5netcdf/pull/%s", "PR%s"),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autosummary_generate = True
autoclass_content = "class"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = f"h5netcdf - {release}"

html_baseurl = "https://h5netcdf.org"

html_context = {
    "github_user": "h5netcdf",
    "github_repo": "h5netcdf",
    "github_version": "main",
    "doc_path": "doc",
}

html_theme_options = {
    "show_toc_level": 2,
    "use_download_button": True,
    "repository_url": "https://github.com/h5netcdf/h5netcdf",
    "repository_branch": "main",
    "path_to_docs": "doc",
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Napoleon settings for docstring processing -------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_special_with_doc = False
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "scalar": ":term:`scalar`",
    "sequence": ":term:`sequence`",
    "callable": ":py:func:`callable`",
    "file-like": ":term:`file-like <file-like object>`",
    "array-like": ":term:`array-like <array_like>`",
    "Path": "~~pathlib.Path",
}

# handle release substition
url = "https://github.com/h5netcdf"

# get version
version_tuple = h5netcdf._version.version_tuple

# is release?
if len(version_tuple) == 3:
    gh_tree_name = f"v{h5netcdf._version.version}"
else:
    # extract git revision
    gh_tree_name = version_tuple[-1].split(".")[0][1:]

rel = f"`{release} <{url}/h5netcdf/tree/{gh_tree_name}>`__"

rst_epilog = ""
rst_epilog += f"""
.. |release| replace:: {rel}
"""
