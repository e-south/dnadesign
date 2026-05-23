from __future__ import annotations

from .campaign_cells import render_campaign_cells
from .details_cells import render_details_cells
from .setup_cells import render_setup_cells
from .visual_cells import render_visual_cells

OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION = "opal.generated_campaign_notebook.v2"


def render_campaign_set_template() -> str:
    """Render the full campaign-set marimo notebook source."""

    return "\n\n".join(
        (
            render_setup_cells(),
            render_campaign_cells(),
            render_visual_cells(),
            render_details_cells(),
        )
    )


__all__ = ["OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION", "render_campaign_set_template"]
