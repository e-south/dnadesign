"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_template/__init__.py

Package exports for OPAL analysis notebook template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .renderer import OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION, render_campaign_notebook

__all__ = [
    "OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION",
    "render_campaign_notebook",
]
