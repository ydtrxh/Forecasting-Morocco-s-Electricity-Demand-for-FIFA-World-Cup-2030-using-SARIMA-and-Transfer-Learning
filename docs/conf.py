# ─────────────────────────────────────────────────────────────────────────────
# conf.py — Sphinx configuration for Morocco 2030 WC Demand Forecast
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

# Make the src/ directory importable for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# ── Project Information ───────────────────────────────────────────────────────
project   = "Morocco 2030 WC Demand Forecast"
copyright = "2026, Younes Chajara and Achraf Oudich — ENSAM Meknès, Filière IATD"
author    = "Younes Chajara and Achraf Oudich"
release   = "1.0.0"
version   = "1.0"

# ── General Configuration ─────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",          # Auto-generate API docs from docstrings
    "sphinx.ext.napoleon",         # NumPy/Google-style docstring support
    "sphinx.ext.mathjax",          # Render LaTeX math equations
    "sphinx.ext.viewcode",         # Add links to source code
    "sphinx.ext.intersphinx",      # Cross-reference external docs
    "sphinx.ext.autosummary",      # Summary tables for modules
    "sphinx.ext.githubpages",      # GitHub Pages compatibility
    "myst_parser",                 # MyST Markdown support
]

# MyST parser settings — enable math and dollarmath
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "colon_fence",
    "deflist",
    "html_admonition",
    "html_image",
    "tasklist",
]
myst_dmath_double_inline = True

# Napoleon settings for NumPy/Google docstrings
napoleon_google_docstring  = True
napoleon_numpy_docstring   = True
napoleon_include_init_with_doc = False
napoleon_use_admonition_for_notes = True

# Autodoc settings
autodoc_default_options = {
    "members":         True,
    "undoc-members":   False,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "description"

# Intersphinx mapping
intersphinx_mapping = {
    "python":  ("https://docs.python.org/3", None),
    "numpy":   ("https://numpy.org/doc/stable", None),
    "pandas":  ("https://pandas.pydata.org/docs", None),
    "torch":   ("https://pytorch.org/docs/stable", None),
}

# Source file suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md":  "markdown",
}

# Root document
root_doc = "index"

# Templates path
templates_path = ["_templates"]

# Patterns to exclude
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ── HTML Output ───────────────────────────────────────────────────────────────
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary":   "#1a6b3c",   # Morocco green
        "color-brand-content":   "#1a6b3c",
        "color-admonition-background": "rgba(26, 107, 60, 0.05)",
    },
    "dark_css_variables": {
        "color-brand-primary":   "#3ecf7a",
        "color-brand-content":   "#3ecf7a",
    },
    "footer_icons": [
        {
            "name":  "GitHub",
            "url":   "https://github.com/your-username/wc2030-morocco-electricity-forecast",
            "html":  """<svg stroke="currentColor" fill="currentColor" stroke-width="0"
                             viewBox="0 0 16 16" height="1em" width="1em"
                             xmlns="http://www.w3.org/2000/svg">
                          <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29
                          6.53 5.47 7.59.4.07.55-.17.55-.38
                          0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01
                          1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                          0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                          2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04
                          2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0
                          3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01
                          1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z">
                          </path>
                       </svg>""",
            "class": "",
        },
    ],
}

html_title         = "Morocco 2030 WC Demand Forecast"
html_short_title   = "Morocco 2030"
html_static_path   = ["_static"]
html_logo          = None
html_favicon       = None
html_show_sourcelink = True
html_copy_source   = True

# ── MathJax ───────────────────────────────────────────────────────────────────
mathjax3_config = {
    "tex": {
        "tags": "ams",
        "macros": {
            "RR": "{\\mathbb{R}}",
            "E":  "{\\mathbb{E}}",
        },
    },
}

# ── LaTeX Output (for PDF) ────────────────────────────────────────────────────
latex_engine = "pdflatex"
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
""",
}
latex_documents = [
    (root_doc, "morocco2030.tex",
     "Morocco 2030 WC Demand Forecast",
     "Younes Chajara and Achraf Oudich", "manual"),
]

# ── Epub Output ───────────────────────────────────────────────────────────────
epub_show_urls = "footnote"
