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
from .baserender_campaign_context import load_notebook_baserender_campaign_context
from .baserender_candidate_catalog import build_notebook_baserender_candidate_catalog
from .baserender_diagnostics import render_notebook_baserender_diagnostic_panel
from .baserender_record_memory import (
    build_notebook_baserender_record_memory_key,
    resolve_notebook_baserender_preferred_record_id,
)
from .baserender_record_selection import (
    build_notebook_selected_baserender_record_sets,
    build_notebook_selected_baserender_records,
    resolve_notebook_baserender_candidate_record,
    resolve_notebook_baserender_selection_batch_scope,
)
from .baserender_records import (
    build_notebook_baserender_label_rows,
    build_notebook_baserender_record_annotation_counts,
    build_notebook_baserender_record_choices,
    build_notebook_baserender_record_choices_with_counts,
    build_notebook_baserender_record_options,
    has_notebook_baserender_record_options,
    load_notebook_baserender_record_row,
    select_notebook_baserender_default_record_id,
)
from .baserender_render import (
    render_notebook_baserender_record,
)
from .baserender_review_bundle import (
    build_notebook_baserender_evidence_bundle,
    build_notebook_baserender_record_controls,
)
from .baserender_selection_scope import (
    build_notebook_baserender_role_control,
    build_notebook_baserender_selection_view_control,
    resolve_notebook_baserender_campaign_model,
    resolve_notebook_baserender_selection_view_id,
)
from .baserender_selector import (
    build_notebook_baserender_selector_model,
    render_notebook_baserender_selector,
)
from .campaign_set_baserender import (
    CAMPAIGN_SET_BASERENDER_SURFACE_KIND,
    build_notebook_collection_baserender_role_choices,
    build_notebook_collection_baserender_role_control,
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
from .label_staging import (
    build_notebook_label_staging_rows,
    discover_label_staging_inputs,
)
from .layered_scatter import (
    build_notebook_layered_scatter_contract,
    build_notebook_layered_scatter_controls,
    filter_notebook_layered_scatter_rows,
    read_notebook_layered_scatter_state,
    render_notebook_layered_scatter_image,
)
from .no_plot_scope import build_notebook_no_plot_scope_rows
from .overview import (
    build_notebook_at_a_glance_rows,
    build_notebook_campaign_header_lines,
    build_notebook_campaign_summary_row,
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
    render_notebook_plot_choice_image,
)
from .project import find_notebook_repo_root, list_notebook_campaign_paths
from .reader_evidence import (
    build_notebook_reader_evidence_artifact_options,
    build_notebook_reader_evidence_artifact_rows,
    build_notebook_reader_evidence_plot_type_options,
    build_notebook_reader_evidence_record_memory_key,
    build_notebook_reader_evidence_rows,
    build_notebook_reader_evidence_surface,
    build_notebook_reader_evidence_visual_choices,
    discover_reader_evidence_artifacts,
    discover_reader_evidence_manifests,
    render_notebook_reader_evidence_artifact_control,
    render_notebook_reader_evidence_artifact_visual,
    render_notebook_reader_evidence_panel,
    render_notebook_reader_evidence_plot_type_control,
    render_notebook_reader_evidence_record_control,
    resolve_notebook_reader_evidence_preferred_record_label,
)
from .review_controls import render_notebook_review_control_surface
from .runs import (
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_change_summary_rows,
    build_notebook_no_run_lines,
    build_notebook_run_options,
    build_notebook_run_summary_lines,
    resolve_notebook_round_default,
)
from .selection_batch import (
    SELECTION_BATCH_SURFACE_KIND,
    build_notebook_selection_batch_choice,
    build_notebook_selection_batch_rows,
    build_notebook_selection_batch_summary_rows,
)
from .selection_overlap import (
    CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND,
    build_notebook_campaign_set_selection_overlap_card_rows,
    build_notebook_campaign_set_selection_overlap_choice,
    build_notebook_campaign_set_selection_overlap_rows,
    render_notebook_campaign_set_selection_overlap_image,
)
from .selection_views import build_notebook_selection_view_options, resolve_notebook_selection_view
from .three_axis_scatter import (
    THREE_AXIS_INTERACTIVE_MODE,
    THREE_AXIS_PUBLICATION_MODE,
    THREE_AXIS_SCATTER_ADAPTER,
    build_notebook_three_axis_scatter_figure,
    render_notebook_three_axis_scatter,
    sample_notebook_three_axis_rows,
)
from .trust import (
    build_notebook_distrust_lines,
    build_notebook_distrust_rows,
    build_notebook_status_line,
    build_notebook_trust_rows,
    build_notebook_validity_lines,
    build_notebook_validity_rows,
)
from .visual_hierarchy import (
    annotate_notebook_visual_choices,
    build_notebook_visual_group_options,
    filter_notebook_visual_choices_by_group,
)
from .visual_panel import render_notebook_visual_panel
from .visual_panel_baserender import build_notebook_baserender_panel_title
from .visual_surface import render_visual_surface_cells
from .zoomable_visual import render_notebook_zoomable_image

__all__ = [
    "CAMPAIGN_SET_SELECTION_OVERLAP_SURFACE_KIND",
    "SELECTION_BATCH_SURFACE_KIND",
    "THREE_AXIS_INTERACTIVE_MODE",
    "THREE_AXIS_PUBLICATION_MODE",
    "THREE_AXIS_SCATTER_ADAPTER",
    "annotate_notebook_visual_choices",
    "build_notebook_artifact_garden_lines",
    "build_notebook_artifact_garden_rows",
    "build_notebook_artifact_garden_summary_rows",
    "build_notebook_at_a_glance_rows",
    "build_notebook_three_axis_scatter_figure",
    "build_notebook_baserender_label_rows",
    "build_notebook_baserender_candidate_catalog",
    "build_notebook_baserender_contract",
    "build_notebook_baserender_contract_rows",
    "build_notebook_baserender_record_annotation_counts",
    "build_notebook_baserender_record_choices",
    "build_notebook_baserender_record_choices_with_counts",
    "build_notebook_baserender_record_options",
    "build_notebook_baserender_record_memory_key",
    "build_notebook_baserender_panel_title",
    "build_notebook_baserender_evidence_bundle",
    "build_notebook_baserender_record_controls",
    "build_notebook_baserender_selector_model",
    "build_notebook_baserender_role_control",
    "build_notebook_baserender_selection_view_control",
    "build_notebook_selected_baserender_record_sets",
    "build_notebook_selected_baserender_records",
    "has_notebook_baserender_record_options",
    "build_notebook_selection_batch_choice",
    "build_notebook_selection_batch_rows",
    "build_notebook_selection_batch_summary_rows",
    "build_notebook_selection_view_options",
    "build_notebook_campaign_header_lines",
    "build_notebook_campaign_summary_row",
    "resolve_notebook_selection_view",
    "build_notebook_collection_baserender_role_choices",
    "build_notebook_collection_baserender_role_control",
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
    "build_notebook_campaign_set_selection_overlap_card_rows",
    "build_notebook_campaign_set_selection_overlap_choice",
    "build_notebook_campaign_set_selection_overlap_rows",
    "build_notebook_distrust_lines",
    "build_notebook_distrust_rows",
    "build_notebook_evidence_rows",
    "build_notebook_label_staging_rows",
    "build_notebook_layered_scatter_contract",
    "build_notebook_layered_scatter_controls",
    "build_notebook_metric_definition_rows",
    "build_notebook_no_run_lines",
    "build_notebook_no_plot_scope_rows",
    "build_notebook_plot_card_rows",
    "build_notebook_plot_inventory_rows",
    "build_notebook_plot_method_sections",
    "build_notebook_plot_method_rows",
    "build_notebook_plot_scope_options",
    "build_notebook_reader_evidence_artifact_rows",
    "build_notebook_reader_evidence_artifact_options",
    "build_notebook_reader_evidence_plot_type_options",
    "build_notebook_reader_evidence_record_memory_key",
    "build_notebook_reader_evidence_rows",
    "build_notebook_reader_evidence_surface",
    "build_notebook_reader_evidence_visual_choices",
    "build_notebook_visual_surface_model",
    "build_notebook_visual_group_options",
    "build_notebook_run_options",
    "build_notebook_run_summary_lines",
    "build_notebook_status_line",
    "build_notebook_trust_rows",
    "build_notebook_validity_lines",
    "build_notebook_validity_rows",
    "compact_notebook_path",
    "CAMPAIGN_SET_BASERENDER_SURFACE_KIND",
    "discover_label_staging_inputs",
    "discover_reader_evidence_artifacts",
    "discover_reader_evidence_manifests",
    "render_notebook_reader_evidence_artifact_control",
    "find_notebook_repo_root",
    "filter_notebook_visual_choices_by_group",
    "filter_notebook_layered_scatter_rows",
    "load_notebook_baserender_record_row",
    "load_notebook_baserender_campaign_context",
    "list_notebook_campaign_paths",
    "render_notebook_baserender_record",
    "render_notebook_baserender_diagnostic_panel",
    "render_notebook_baserender_selector",
    "render_notebook_campaign_set_metric_comparison_image",
    "render_notebook_campaign_set_vector_heatmap_comparison_image",
    "render_notebook_campaign_set_plot_gallery_image",
    "render_notebook_plot_choice_image",
    "render_notebook_reader_evidence_artifact_visual",
    "render_notebook_reader_evidence_panel",
    "render_notebook_reader_evidence_plot_type_control",
    "render_notebook_reader_evidence_record_control",
    "resolve_notebook_reader_evidence_preferred_record_label",
    "render_notebook_layered_scatter_image",
    "read_notebook_layered_scatter_state",
    "render_notebook_review_control_surface",
    "render_notebook_campaign_set_selection_overlap_image",
    "render_notebook_visual_panel",
    "render_notebook_zoomable_image",
    "render_notebook_three_axis_scatter",
    "render_visual_surface_cells",
    "resolve_notebook_round_default",
    "resolve_notebook_baserender_selection_view_id",
    "resolve_notebook_baserender_candidate_record",
    "resolve_notebook_baserender_selection_batch_scope",
    "resolve_notebook_baserender_preferred_record_id",
    "resolve_notebook_baserender_campaign_model",
    "select_notebook_baserender_default_record_id",
    "sample_notebook_three_axis_rows",
    "select_notebook_plot_scope",
]
