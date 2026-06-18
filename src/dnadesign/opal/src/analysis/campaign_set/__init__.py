"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign_set/__init__.py

Campaign-set semantic view-model helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .visual_kinds import (
    COLLECTION_PLOT_GALLERY_SURFACE_KIND,
    COLLECTION_VECTOR_HEATMAP_SURFACE_KIND,
    COLLECTION_VECTOR_MSE_SURFACE_KIND,
    COLLECTION_VISUAL_SURFACE_KIND,
    collection_visual_source_plot_kind_for_view_kind,
    collection_visual_surface_kind_for_view_kind,
    list_collection_comparison_view_kinds,
    list_collection_visual_surface_kinds,
    require_collection_visual_surface_kind,
)
from .visuals import (
    CAMPAIGN_SET_VISUAL_MODEL_SCHEMA_VERSION,
    build_campaign_set_collection_visual_model,
)

__all__ = [
    "CAMPAIGN_SET_VISUAL_MODEL_SCHEMA_VERSION",
    "COLLECTION_PLOT_GALLERY_SURFACE_KIND",
    "COLLECTION_VISUAL_SURFACE_KIND",
    "COLLECTION_VECTOR_HEATMAP_SURFACE_KIND",
    "COLLECTION_VECTOR_MSE_SURFACE_KIND",
    "build_campaign_set_collection_visual_model",
    "collection_visual_source_plot_kind_for_view_kind",
    "collection_visual_surface_kind_for_view_kind",
    "list_collection_comparison_view_kinds",
    "list_collection_visual_surface_kinds",
    "require_collection_visual_surface_kind",
]
