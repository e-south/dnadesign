"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_set_template/cells.py

Notebook-set template builders for cells OPAL analysis notebook set template.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .baserender_cells import render_baserender_cells
from .campaign_cells import render_campaign_cells
from .collection_cells import render_collection_cells
from .details_cells import render_details_cells
from .setup_cells import render_setup_cells
from .visual_cells import render_visual_cells

OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION = "opal.generated_campaign_review_notebook.v5"


def render_campaign_set_template() -> str:
    """Render the full campaign-set marimo notebook source."""

    return "\n".join(
        (
            render_setup_cells(),
            render_campaign_cells(),
            render_collection_cells(),
            render_baserender_cells(),
            render_visual_cells(),
            render_details_cells(),
        )
    )


__all__ = ["OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION", "render_campaign_set_template"]
