"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish_inventory.py

Normalized-payload, inventory, and manifest builders for YIU publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.cruncher.yiu.bundle_models import (
    PayloadBundleManifest,
    PayloadViewEntry,
    PayloadVisualInventory,
    payload_summary_from_normalized,
)
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.publish_layout import PayloadBundleLayout, build_published_artifacts
from dnadesign.cruncher.yiu.spec_models import YiuPayloadRenderingSpec


def _published_plot_artifact_path(spec: YiuPayloadRenderingSpec) -> str | None:
    if spec.output.published_plot_path is None:
        return None
    return str(spec.output.published_plot_path)


def build_normalized_payload_dump(
    *,
    spec: YiuPayloadRenderingSpec,
    normalized: NormalizedPayload,
    layout: PayloadBundleLayout,
) -> dict[str, object]:
    published_artifacts = build_published_artifacts(
        layout=layout,
        published_plot_artifact_path=_published_plot_artifact_path(spec),
    )
    return normalized.model_copy(update={"published_artifacts": published_artifacts}).model_dump(mode="json")


def build_payload_visual_inventory(
    *,
    spec: YiuPayloadRenderingSpec,
    normalized: NormalizedPayload,
    layout: PayloadBundleLayout,
    view_entries: list[PayloadViewEntry],
) -> PayloadVisualInventory:
    return PayloadVisualInventory(
        spec_name=spec.yiu.name,
        input_kind=normalized.input_kind,
        view_count=len(view_entries),
        render_count=0,
        render_status="not_requested",
        composite_render_artifact_path=layout.relative_artifact_path(layout.composite_render_path),
        published_plot_artifact_path=_published_plot_artifact_path(spec),
        pwm_effective=normalized.motif_context.effective,
        payload_view_requires_motif_layers=normalized.motif_context.effective,
        views=view_entries,
    )


def build_payload_bundle_manifest(
    *,
    normalized: NormalizedPayload,
    inventory: PayloadVisualInventory,
) -> PayloadBundleManifest:
    return PayloadBundleManifest(
        spec_name=inventory.spec_name,
        provenance=normalized.source_provenance,
        payload_view_requires_motif_layers=inventory.payload_view_requires_motif_layers,
        view_contracts=inventory.views,
        composite_render_artifact_path=inventory.composite_render_artifact_path,
        published_plot_artifact_path=inventory.published_plot_artifact_path,
        render_status=inventory.render_status,
        **payload_summary_from_normalized(normalized),
    )


__all__ = [
    "build_normalized_payload_dump",
    "build_payload_bundle_manifest",
    "build_payload_visual_inventory",
]
