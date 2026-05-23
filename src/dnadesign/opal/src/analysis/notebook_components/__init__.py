"""
Reusable generated-cell components for OPAL marimo notebooks.
"""

from __future__ import annotations

from ._support import compact_notebook_path
from .artifacts import (
    build_notebook_artifact_garden_lines,
    build_notebook_artifact_garden_rows,
    build_notebook_artifact_garden_summary_rows,
)
from .baserender import (
    build_notebook_baserender_contract,
    build_notebook_baserender_contract_rows,
)
from .baserender_records import (
    build_notebook_baserender_label_rows,
    build_notebook_baserender_record_options,
    load_notebook_baserender_record_row,
)
from .baserender_render import (
    render_notebook_baserender_record,
)
from .overview import (
    build_notebook_at_a_glance_rows,
    build_notebook_campaign_header_lines,
    build_notebook_campaign_summary_row,
    build_notebook_distrust_lines,
    build_notebook_distrust_rows,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_status_line,
    build_notebook_trust_rows,
    build_notebook_validity_lines,
    build_notebook_validity_rows,
)
from .plot_scopes import build_notebook_plot_scope_options, select_notebook_plot_scope
from .plots import (
    build_notebook_plot_card_rows,
    build_notebook_plot_inventory_rows,
    build_notebook_plot_method_rows,
    build_notebook_plot_method_sections,
    build_notebook_visual_surface_model,
)
from .project import find_notebook_repo_root, list_notebook_campaign_paths
from .runs import (
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_change_summary_rows,
    build_notebook_no_run_lines,
    build_notebook_run_options,
    build_notebook_run_summary_lines,
    resolve_notebook_round_default,
)
from .visual_surface import render_visual_surface_cells

__all__ = [
    "build_notebook_artifact_garden_lines",
    "build_notebook_artifact_garden_rows",
    "build_notebook_artifact_garden_summary_rows",
    "build_notebook_at_a_glance_rows",
    "build_notebook_baserender_label_rows",
    "build_notebook_baserender_contract",
    "build_notebook_baserender_contract_rows",
    "build_notebook_baserender_record_options",
    "build_notebook_campaign_header_lines",
    "build_notebook_campaign_summary_row",
    "build_notebook_change_lines",
    "build_notebook_change_rows",
    "build_notebook_change_summary_rows",
    "build_notebook_distrust_lines",
    "build_notebook_distrust_rows",
    "build_notebook_evidence_rows",
    "build_notebook_metric_definition_rows",
    "build_notebook_no_run_lines",
    "build_notebook_plot_card_rows",
    "build_notebook_plot_inventory_rows",
    "build_notebook_plot_method_sections",
    "build_notebook_plot_method_rows",
    "build_notebook_plot_scope_options",
    "build_notebook_visual_surface_model",
    "build_notebook_run_options",
    "build_notebook_run_summary_lines",
    "build_notebook_status_line",
    "build_notebook_trust_rows",
    "build_notebook_validity_lines",
    "build_notebook_validity_rows",
    "compact_notebook_path",
    "find_notebook_repo_root",
    "load_notebook_baserender_record_row",
    "list_notebook_campaign_paths",
    "render_notebook_baserender_record",
    "render_visual_surface_cells",
    "resolve_notebook_round_default",
    "select_notebook_plot_scope",
]
