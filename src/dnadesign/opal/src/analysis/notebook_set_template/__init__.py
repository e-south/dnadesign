"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/__init__.py

Package exports for OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .cells import OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION
from .renderer import render_campaign_set_notebook

__all__ = ["OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION", "render_campaign_set_notebook"]
