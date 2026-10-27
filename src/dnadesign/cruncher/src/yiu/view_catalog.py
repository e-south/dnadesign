"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/view_catalog.py

Canonical view registry and render-job planning for payload-centric YIU bundles.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_registry import canonical_payload_view_definitions, payload_view_definition
from dnadesign.cruncher.yiu.view_styles import get_yiu_style_profile

if TYPE_CHECKING:
    from dnadesign.cruncher.yiu.publish_layout import PayloadBundleLayout


def build_payload_view_entries(
    *,
    layout: PayloadBundleLayout,
    normalized: NormalizedPayload,
) -> list[PayloadViewEntry]:
    composite_render_path = layout.relative_artifact_path(layout.composite_render_path)
    view_paths = {
        "payload": layout.relative_artifact_path(layout.payload_view_path),
        "split_payload": layout.relative_artifact_path(layout.split_payload_view_path),
        "assembled_payload": layout.relative_artifact_path(layout.assembled_payload_view_path),
    }
    motif_layers_required = {"payload": normalized.motif_context.effective}
    entries: list[PayloadViewEntry] = []
    for view in canonical_payload_view_definitions():
        definition = payload_view_definition(view.view_id)
        style_profile = get_yiu_style_profile(view.view_id)
        if style_profile.direction_name != definition.visual_direction:
            raise ValueError(
                f"YIU visual-direction drift for {view.view_id!r}: "
                f"{style_profile.direction_name!r} != {definition.visual_direction!r}"
            )
        entries.append(
            PayloadViewEntry(
                view_id=view.view_id,
                visual_direction=definition.visual_direction,
                contract_kind=view.contract_kind,
                schema_version=1,
                input_kind=view.input_kind,
                view_contract_path=view_paths[view.view_id],
                render_artifact_path=composite_render_path,
                renderer_kind=view.renderer_kind,
                style_overrides=style_profile.style_overrides,
                motif_layers_required=motif_layers_required.get(view.view_id, False),
            )
        )
    return entries


def build_render_job_payload(*, entry: PayloadViewEntry) -> dict[str, object]:
    return {
        "version": 4,
        "bundle": {"path": f"../debug/rerenders/{entry.view_id}.render-v1"},
        "input": {
            "kind": entry.input_kind,
            "path": f"../{entry.view_contract_path}",
            "adapter": {"kind": entry.contract_kind},
            "alphabet": "iupac_dna",
        },
        "render": {
            "renderer": entry.renderer_kind,
            "style": {"preset": entry.style_preset, "overrides": entry.style_overrides},
        },
        "outputs": [{"kind": "images", "path": f"{entry.view_id}.pdf", "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True},
    }


__all__ = [
    "build_payload_view_entries",
    "build_render_job_payload",
    "canonical_payload_view_definitions",
]
