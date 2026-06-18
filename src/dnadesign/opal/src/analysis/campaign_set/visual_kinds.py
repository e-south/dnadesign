"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/campaign_set/visual_kinds.py

Collection visual-kind contracts for campaign-set notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.utils import ExitCodes, OpalError

COLLECTION_VISUAL_SURFACE_KIND = "campaign_set_metric_comparison"
COLLECTION_VECTOR_MSE_SURFACE_KIND = "campaign_set_vector_reference_mse_comparison"
COLLECTION_VECTOR_HEATMAP_SURFACE_KIND = "campaign_set_vector_heatmap_comparison"
COLLECTION_PLOT_GALLERY_SURFACE_KIND = "campaign_set_plot_gallery"


@dataclass(frozen=True)
class CollectionVisualKindSpec:
    """Contract linking collection manifest view kinds to notebook visual surfaces."""

    view_kind: str
    surface_kind: str
    source_plot_kind: str


COLLECTION_VISUAL_KIND_SPECS: dict[str, CollectionVisualKindSpec] = {
    "metric_over_rounds_comparison": CollectionVisualKindSpec(
        view_kind="metric_over_rounds_comparison",
        surface_kind=COLLECTION_VISUAL_SURFACE_KIND,
        source_plot_kind="metric_over_rounds",
    ),
    "paired_plot_gallery": CollectionVisualKindSpec(
        view_kind="paired_plot_gallery",
        surface_kind=COLLECTION_PLOT_GALLERY_SURFACE_KIND,
        source_plot_kind="vector_summary_heatmap",
    ),
    "vector_heatmap_comparison": CollectionVisualKindSpec(
        view_kind="vector_heatmap_comparison",
        surface_kind=COLLECTION_VECTOR_HEATMAP_SURFACE_KIND,
        source_plot_kind="vector_summary_heatmap",
    ),
    "vector_reference_mse_over_rounds_comparison": CollectionVisualKindSpec(
        view_kind="vector_reference_mse_over_rounds_comparison",
        surface_kind=COLLECTION_VECTOR_MSE_SURFACE_KIND,
        source_plot_kind="vector_summary_heatmap",
    ),
}


def list_collection_comparison_view_kinds() -> frozenset[str]:
    return frozenset(COLLECTION_VISUAL_KIND_SPECS)


def list_collection_visual_surface_kinds() -> frozenset[str]:
    return frozenset(spec.surface_kind for spec in COLLECTION_VISUAL_KIND_SPECS.values())


def collection_visual_kind_spec(view_kind: str) -> CollectionVisualKindSpec:
    text = str(view_kind or "").strip()
    spec = COLLECTION_VISUAL_KIND_SPECS.get(text)
    if spec is None:
        raise OpalError(
            f"Unsupported collection comparison view kind: {text!r}. "
            f"Expected one of {sorted(COLLECTION_VISUAL_KIND_SPECS)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return spec


def collection_visual_surface_kind_for_view_kind(view_kind: str) -> str:
    return collection_visual_kind_spec(view_kind).surface_kind


def collection_visual_source_plot_kind_for_view_kind(view_kind: str) -> str:
    return collection_visual_kind_spec(view_kind).source_plot_kind


def require_collection_visual_surface_kind(surface_kind: str, *, field: str = "surface_kind") -> str:
    text = str(surface_kind or "").strip()
    allowed = list_collection_visual_surface_kinds()
    if text not in allowed:
        raise OpalError(
            f"Unsupported collection visual {field}: {text!r}. Expected one of {sorted(allowed)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return text


__all__ = [
    "COLLECTION_PLOT_GALLERY_SURFACE_KIND",
    "COLLECTION_VECTOR_HEATMAP_SURFACE_KIND",
    "COLLECTION_VECTOR_MSE_SURFACE_KIND",
    "COLLECTION_VISUAL_KIND_SPECS",
    "COLLECTION_VISUAL_SURFACE_KIND",
    "CollectionVisualKindSpec",
    "collection_visual_kind_spec",
    "collection_visual_source_plot_kind_for_view_kind",
    "collection_visual_surface_kind_for_view_kind",
    "list_collection_comparison_view_kinds",
    "list_collection_visual_surface_kinds",
    "require_collection_visual_surface_kind",
]
