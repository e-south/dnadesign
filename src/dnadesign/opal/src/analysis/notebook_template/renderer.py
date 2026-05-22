"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/notebook_template/renderer.py

Renders marimo notebook templates for OPAL campaigns. Generates scaffolded
notebooks with campaign context and data previews.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..notebook_components import render_plot_gallery_cells
from .source import render_notebook_source

OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION = "opal.generated_campaign_notebook.v1"


def render_campaign_notebook(config_path: Path, *, round_selector: str) -> str:
    """
    Render a marimo notebook template tied to a campaign.
    """
    try:
        import marimo as _marimo
    except Exception:
        _marimo = None
    if _marimo is None:
        marimo_version = "unknown"
    else:
        marimo_version = getattr(_marimo, "__version__", "unknown")

    template = render_notebook_source(plot_gallery_cells=render_plot_gallery_cells())

    return (
        template.replace("__CONFIG_PATH__", repr(str(config_path)))
        .replace("__DEFAULT_ROUND__", repr(str(round_selector)))
        .replace("__OPAL_NOTEBOOK_TEMPLATE_SCHEMA__", OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION)
        .replace("__GENERATED_WITH__", str(marimo_version))
        + "\n"
    )
