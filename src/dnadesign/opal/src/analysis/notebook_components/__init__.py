"""
Reusable generated-cell components for OPAL marimo notebooks.
"""

from __future__ import annotations

from .artifacts import build_notebook_artifact_garden_lines, build_notebook_artifact_garden_rows
from .overview import (
    build_notebook_at_a_glance_lines,
    build_notebook_campaign_summary_row,
    build_notebook_distrust_lines,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_validity_lines,
)
from .plots import (
    build_notebook_plot_card_lines,
    build_notebook_plot_gallery_model,
    render_plot_gallery_cells,
)
from .runs import (
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_no_run_lines,
    build_notebook_run_options,
    build_notebook_run_summary_lines,
    resolve_notebook_round_default,
)

__all__ = [
    "build_notebook_artifact_garden_lines",
    "build_notebook_artifact_garden_rows",
    "build_notebook_at_a_glance_lines",
    "build_notebook_campaign_summary_row",
    "build_notebook_change_lines",
    "build_notebook_change_rows",
    "build_notebook_distrust_lines",
    "build_notebook_evidence_rows",
    "build_notebook_metric_definition_rows",
    "build_notebook_no_run_lines",
    "build_notebook_plot_card_lines",
    "build_notebook_plot_gallery_model",
    "build_notebook_run_options",
    "build_notebook_run_summary_lines",
    "build_notebook_validity_lines",
    "render_plot_gallery_cells",
    "resolve_notebook_round_default",
]
