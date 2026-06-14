"""Renderer registry for materialized campaign-set collection visuals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from ..analysis.campaign_set import (
    COLLECTION_PLOT_GALLERY_SURFACE_KIND,
    COLLECTION_VECTOR_HEATMAP_SURFACE_KIND,
    COLLECTION_VECTOR_MSE_SURFACE_KIND,
    COLLECTION_VISUAL_SURFACE_KIND,
    require_collection_visual_surface_kind,
)
from ..analysis.notebook_components import (
    build_notebook_campaign_set_metric_comparison_rows,
    build_notebook_campaign_set_plot_gallery_items,
    build_notebook_campaign_set_vector_heatmap_rows,
    build_notebook_campaign_set_vector_reference_mse_rows,
    render_notebook_campaign_set_metric_comparison_image,
    render_notebook_campaign_set_plot_gallery_image,
    render_notebook_campaign_set_vector_heatmap_comparison_image,
)
from ..core.utils import ExitCodes, OpalError


@dataclass(frozen=True)
class CollectionVisualRenderResult:
    rows: list[Mapping[str, Any]]
    rendered: dict[str, Any] | None
    input_paths: list[Path]


CollectionVisualRenderer = Callable[
    [list[Mapping[str, Any]], Mapping[str, Any], Path],
    CollectionVisualRenderResult,
]


def render_collection_visual_artifact(
    campaigns: list[Mapping[str, Any]],
    *,
    visual: Mapping[str, Any],
    media_path: Path,
) -> CollectionVisualRenderResult:
    """Render one materialized collection visual through the registered surface kind."""

    surface_kind = require_collection_visual_surface_kind(str(visual.get("surface_kind") or ""))
    try:
        renderer = COLLECTION_VISUAL_RENDERERS[surface_kind]
    except KeyError as exc:
        raise OpalError(
            f"Unsupported collection visual surface_kind: {surface_kind!r}",
            ExitCodes.CONTRACT_VIOLATION,
        ) from exc
    return renderer(campaigns, visual, media_path)


def _render_metric_comparison(
    campaigns: list[Mapping[str, Any]],
    visual: Mapping[str, Any],
    media_path: Path,
) -> CollectionVisualRenderResult:
    del media_path
    rows = build_notebook_campaign_set_metric_comparison_rows(
        campaigns,
        plot_name=str(visual["source_plot_name"]),
        group_key=str(visual["group_key"]),
        summary=str(visual["summary"]),
        relationship=visual,
    )
    rows = _filter_metric_rows(rows, visual=visual)
    rendered = render_notebook_campaign_set_metric_comparison_image(
        rows,
        title=str(visual.get("title") or visual.get("label") or visual.get("id")),
        group_key=str(visual["group_key"]),
        interval_kind=str(visual.get("interval_kind") or "none"),
        confidence_level=visual.get("confidence_level") if visual.get("confidence_level") is not None else None,
        interpretation_note=str(visual.get("interpretation_note") or ""),
    )
    return CollectionVisualRenderResult(rows=rows, rendered=rendered, input_paths=_source_tidy_paths(rows))


def _render_vector_reference_mse(
    campaigns: list[Mapping[str, Any]],
    visual: Mapping[str, Any],
    media_path: Path,
) -> CollectionVisualRenderResult:
    del media_path
    rows = build_notebook_campaign_set_vector_reference_mse_rows(
        campaigns,
        plot_name=str(visual["source_plot_name"]),
        group_key=str(visual["group_key"]),
        relationship=visual,
        cohort=str(visual.get("cohort") or "selected"),
    )
    rows = _filter_metric_rows(rows, visual=visual)
    rendered = render_notebook_campaign_set_metric_comparison_image(
        rows,
        title=str(visual.get("title") or visual.get("label") or visual.get("id")),
        group_key=str(visual["group_key"]),
        interval_kind=str(visual.get("interval_kind") or "none"),
        confidence_level=visual.get("confidence_level") if visual.get("confidence_level") is not None else None,
        interpretation_note=str(visual.get("interpretation_note") or ""),
    )
    return CollectionVisualRenderResult(rows=rows, rendered=rendered, input_paths=_source_tidy_paths(rows))


def _render_plot_gallery(
    campaigns: list[Mapping[str, Any]],
    visual: Mapping[str, Any],
    media_path: Path,
) -> CollectionVisualRenderResult:
    del media_path
    rows = build_notebook_campaign_set_plot_gallery_items(
        campaigns,
        plot_name=str(visual["source_plot_name"]),
        plot_kind=str(visual["source_plot_kind"]),
        group_key=str(visual["group_key"]),
        relationship=visual,
    )
    rendered = render_notebook_campaign_set_plot_gallery_image(
        rows,
        title=str(visual.get("title") or visual.get("label") or visual.get("id")),
        group_key=str(visual["group_key"]),
    )
    return CollectionVisualRenderResult(rows=rows, rendered=rendered, input_paths=_source_media_paths(rows))


def _render_vector_heatmap(
    campaigns: list[Mapping[str, Any]],
    visual: Mapping[str, Any],
    media_path: Path,
) -> CollectionVisualRenderResult:
    del media_path
    rows = build_notebook_campaign_set_vector_heatmap_rows(
        campaigns,
        plot_name=str(visual["source_plot_name"]),
        group_key=str(visual["group_key"]),
        relationship=visual,
        cohort=str(visual.get("cohort") or "selected"),
    )
    rendered = render_notebook_campaign_set_vector_heatmap_comparison_image(
        rows,
        title=str(visual.get("title") or visual.get("label") or visual.get("id")),
        group_key=str(visual["group_key"]),
        interval_kind=str(visual.get("interval_kind") or "none"),
        interpretation_note=str(visual.get("interpretation_note") or ""),
    )
    return CollectionVisualRenderResult(rows=rows, rendered=rendered, input_paths=_source_tidy_paths(rows))


def _filter_metric_rows(rows: list[dict[str, Any]], *, visual: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("metric") or "") == str(visual["metric"])
        and str(row.get("cohort") or "") == str(visual["cohort"])
    ]


def _source_tidy_paths(rows: list[Mapping[str, Any]]) -> list[Path]:
    paths = sorted({str(row.get("tidy_csv") or "") for row in rows if row.get("tidy_csv")})
    return [Path(path) for path in paths]


def _source_media_paths(rows: list[Mapping[str, Any]]) -> list[Path]:
    paths = sorted({str(row.get("media_path") or "") for row in rows if row.get("media_path")})
    return [Path(path) for path in paths]


COLLECTION_VISUAL_RENDERERS: dict[str, CollectionVisualRenderer] = {
    COLLECTION_VISUAL_SURFACE_KIND: _render_metric_comparison,
    COLLECTION_VECTOR_MSE_SURFACE_KIND: _render_vector_reference_mse,
    COLLECTION_PLOT_GALLERY_SURFACE_KIND: _render_plot_gallery,
    COLLECTION_VECTOR_HEATMAP_SURFACE_KIND: _render_vector_heatmap,
}


__all__ = [
    "COLLECTION_VISUAL_RENDERERS",
    "CollectionVisualRenderResult",
    "render_collection_visual_artifact",
]
