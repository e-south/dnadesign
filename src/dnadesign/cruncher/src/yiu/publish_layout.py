"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish_layout.py

Bundle layout and view-entry helpers for YIU payload publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.yiu.bundle_models import PayloadViewEntry
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_contracts import build_yiu_style_overrides


@dataclass(frozen=True)
class PayloadBundleLayout:
    bundle_dir: Path
    render_jobs_dir: Path
    composite_render_path: Path
    payload_view_path: Path
    split_payload_view_path: Path
    assembled_payload_view_path: Path
    normalized_payload_path: Path
    manifest_path: Path
    inventory_path: Path

    def relative_artifact_path(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.bundle_dir.resolve()))


def resolve_payload_bundle_layout(bundle_dir: Path) -> PayloadBundleLayout:
    resolved_bundle_dir = bundle_dir.resolve()
    return PayloadBundleLayout(
        bundle_dir=resolved_bundle_dir,
        render_jobs_dir=resolved_bundle_dir / "baserender_jobs",
        composite_render_path=resolved_bundle_dir / "payload_views.pdf",
        payload_view_path=resolved_bundle_dir / "payload_view.json",
        split_payload_view_path=resolved_bundle_dir / "split_payload_view.json",
        assembled_payload_view_path=resolved_bundle_dir / "assembled_payload_view.json",
        normalized_payload_path=resolved_bundle_dir / "normalized_payload.json",
        manifest_path=resolved_bundle_dir / "bundle_manifest.json",
        inventory_path=resolved_bundle_dir / "visual_inventory.json",
    )


def build_published_artifacts(
    *,
    layout: PayloadBundleLayout,
    published_plot_artifact_path: str | None,
) -> dict[str, str]:
    artifacts = {
        "normalized_payload": layout.relative_artifact_path(layout.normalized_payload_path),
        "bundle_manifest": layout.relative_artifact_path(layout.manifest_path),
        "visual_inventory": layout.relative_artifact_path(layout.inventory_path),
        "payload_view": layout.relative_artifact_path(layout.payload_view_path),
        "split_payload_view": layout.relative_artifact_path(layout.split_payload_view_path),
        "assembled_payload_view": layout.relative_artifact_path(layout.assembled_payload_view_path),
        "payload_views_pdf": layout.relative_artifact_path(layout.composite_render_path),
    }
    if published_plot_artifact_path is not None:
        artifacts["published_plot_pdf"] = published_plot_artifact_path
    return artifacts


def build_payload_view_entries(
    *,
    layout: PayloadBundleLayout,
    normalized: NormalizedPayload,
) -> list[PayloadViewEntry]:
    composite_render_path = layout.relative_artifact_path(layout.composite_render_path)
    motif_layers_required = normalized.motif_context.effective
    return [
        PayloadViewEntry(
            view_id="payload",
            contract_kind="yiu_payload_visual_v1",
            schema_version=1,
            input_kind="json",
            view_contract_path=layout.relative_artifact_path(layout.payload_view_path),
            render_artifact_path=composite_render_path,
            renderer_kind="nucleotide_evidence_map",
            style_overrides=build_yiu_style_overrides("payload"),
            motif_layers_required=motif_layers_required,
        ),
        PayloadViewEntry(
            view_id="split_payload",
            contract_kind="sequence_evidence_map_v1",
            schema_version=1,
            input_kind="jsonl",
            view_contract_path=layout.relative_artifact_path(layout.split_payload_view_path),
            render_artifact_path=composite_render_path,
            renderer_kind="sequence_rows",
            style_overrides=build_yiu_style_overrides("split_payload"),
            motif_layers_required=False,
        ),
        PayloadViewEntry(
            view_id="assembled_payload",
            contract_kind="sequence_evidence_map_v1",
            schema_version=1,
            input_kind="json",
            view_contract_path=layout.relative_artifact_path(layout.assembled_payload_view_path),
            render_artifact_path=composite_render_path,
            renderer_kind="nucleotide_evidence_map",
            style_overrides=build_yiu_style_overrides("assembled_payload"),
            motif_layers_required=False,
        ),
    ]


def build_render_job_payload(*, entry: PayloadViewEntry) -> dict[str, object]:
    return {
        "version": 3,
        "results_root": "..",
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
        "outputs": [{"kind": "images", "path": f"../debug/rerenders/{entry.view_id}.pdf", "fmt": "pdf"}],
        "run": {"strict": True, "fail_on_skips": True, "emit_report": False},
    }


__all__ = [
    "PayloadBundleLayout",
    "build_payload_view_entries",
    "build_published_artifacts",
    "build_render_job_payload",
    "resolve_payload_bundle_layout",
]
