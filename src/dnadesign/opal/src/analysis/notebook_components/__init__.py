"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/__init__.py

Reusable generated-cell components for OPAL marimo notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
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
    build_notebook_baserender_record_annotation_counts,
    build_notebook_baserender_record_choices,
    build_notebook_baserender_record_choices_with_counts,
    build_notebook_baserender_record_options,
    build_notebook_selected_baserender_record_ids,
    load_notebook_baserender_record_row,
    select_notebook_baserender_default_record_id,
)
from .baserender_render import (
    render_notebook_baserender_record,
)
from .campaign_set_baserender import (
    CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
    build_notebook_collection_baserender_role_choices,
)
from .campaign_set_comparison import render_notebook_campaign_set_metric_comparison_image
from .campaign_set_gallery import (
    build_notebook_campaign_set_plot_gallery_items,
    render_notebook_campaign_set_plot_gallery_image,
)
from .campaign_set_metric_rows import build_notebook_campaign_set_metric_comparison_rows
from .campaign_set_vector import build_notebook_campaign_set_vector_reference_mse_rows
from .campaign_set_vector_heatmap import (
    build_notebook_campaign_set_vector_heatmap_rows,
    render_notebook_campaign_set_vector_heatmap_comparison_image,
)
from .campaign_set_visuals import (
    build_notebook_collection_set_choices,
    build_notebook_collection_visual_card_rows,
    build_notebook_collection_visual_choices,
    build_notebook_collection_visual_description,
)
from .evidence import (
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
)
from .no_plot_scope import build_notebook_no_plot_scope_rows
from .overview import (
    build_notebook_at_a_glance_rows,
    build_notebook_campaign_header_lines,
    build_notebook_campaign_summary_row,
    build_notebook_distrust_lines,
    build_notebook_distrust_rows,
    build_notebook_status_line,
    build_notebook_trust_rows,
    build_notebook_validity_lines,
    build_notebook_validity_rows,
)
from .plot_method import (
    build_notebook_plot_card_rows,
    build_notebook_plot_method_rows,
    build_notebook_plot_method_sections,
)
from .plot_scopes import build_notebook_plot_scope_options, select_notebook_plot_scope
from .plots import (
    build_notebook_plot_inventory_rows,
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
    "build_notebook_baserender_record_annotation_counts",
    "build_notebook_baserender_record_choices",
    "build_notebook_baserender_record_choices_with_counts",
    "build_notebook_baserender_record_options",
    "build_notebook_selected_baserender_record_ids",
    "build_notebook_campaign_header_lines",
    "build_notebook_campaign_summary_row",
    "build_notebook_collection_baserender_role_choices",
    "build_notebook_campaign_set_metric_comparison_rows",
    "build_notebook_campaign_set_plot_gallery_items",
    "build_notebook_campaign_set_vector_reference_mse_rows",
    "build_notebook_campaign_set_vector_heatmap_rows",
    "build_notebook_collection_set_choices",
    "build_notebook_collection_visual_card_rows",
    "build_notebook_collection_visual_choices",
    "build_notebook_collection_visual_description",
    "build_notebook_change_lines",
    "build_notebook_change_rows",
    "build_notebook_change_summary_rows",
    "build_notebook_distrust_lines",
    "build_notebook_distrust_rows",
    "build_notebook_evidence_rows",
    "build_notebook_metric_definition_rows",
    "build_notebook_no_run_lines",
    "build_notebook_no_plot_scope_rows",
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
    "CAMPAIGN_SET_BASERENDER_SURFACE_KIND",
    "find_notebook_repo_root",
    "load_notebook_baserender_record_row",
    "list_notebook_campaign_paths",
    "render_notebook_baserender_record",
    "render_notebook_campaign_set_metric_comparison_image",
    "render_notebook_campaign_set_vector_heatmap_comparison_image",
    "render_notebook_campaign_set_plot_gallery_image",
    "render_visual_surface_cells",
    "resolve_notebook_round_default",
    "select_notebook_baserender_default_record_id",
    "select_notebook_plot_scope",
]
