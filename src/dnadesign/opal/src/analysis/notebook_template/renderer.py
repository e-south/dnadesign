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

from ..notebook_set_template import render_campaign_set_notebook
from ..notebook_set_template.cells import OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION


def render_campaign_notebook(config_path: Path, *, round_selector: str, run_id: str | None = None) -> str:
    """
    Render the canonical OPAL campaign notebook for one campaign.

    The single-campaign surface intentionally delegates to the same template as
    campaign-set notebooks. A one-campaign notebook is just a campaign-set
    notebook with one selectable campaign, which keeps progress, plot, and
    evidence behavior from drifting across notebook entrypoints.
    """
    return render_campaign_set_notebook([Path(config_path)], round_selector=round_selector, run_id=run_id)


__all__ = ["OPAL_NOTEBOOK_TEMPLATE_SCHEMA_VERSION", "render_campaign_notebook"]
