"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish_layout.py

Bundle layout helpers for YIU payload publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


__all__ = [
    "PayloadBundleLayout",
    "build_published_artifacts",
    "resolve_payload_bundle_layout",
]
