"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/yiu/publish_io.py

Filesystem write helpers for payload-centric YIU publication.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.artifacts.atomic_write import atomic_write_yaml
from dnadesign.cruncher.yiu.bundle_models import PayloadBundleManifest, PayloadViewEntry, PayloadVisualInventory
from dnadesign.cruncher.yiu.bundle_state import persist_bundle_models
from dnadesign.cruncher.yiu.bundle_summary import YiuBundleSummary
from dnadesign.cruncher.yiu.publish_layout import PayloadBundleLayout
from dnadesign.cruncher.yiu.view_catalog import build_render_job_payload
from dnadesign.cruncher.yiu.view_io import write_json_payload, write_jsonl_rows


def remove_stale_payload_bundle_artifacts(*, layout: PayloadBundleLayout) -> None:
    legacy_paths = [layout.bundle_dir / "split_payload_view.json"]
    for path in legacy_paths:
        if path != layout.split_payload_view_path and path.exists():
            Path(path).unlink()


def write_payload_bundle_views(
    *,
    layout: PayloadBundleLayout,
    payload_contract: dict[str, object],
    split_payload_rows: list[dict[str, object]],
    assembled_payload_contract: dict[str, object],
) -> None:
    remove_stale_payload_bundle_artifacts(layout=layout)
    write_json_payload(layout.payload_view_path, payload_contract)
    write_jsonl_rows(layout.split_payload_view_path, split_payload_rows)
    write_json_payload(layout.assembled_payload_view_path, assembled_payload_contract)


def write_normalized_payload_dump(
    *,
    layout: PayloadBundleLayout,
    normalized_payload_dump: dict[str, object],
) -> None:
    write_json_payload(layout.normalized_payload_path, normalized_payload_dump)


def write_payload_bundle_summary(
    *,
    layout: PayloadBundleLayout,
    bundle_summary: YiuBundleSummary,
) -> None:
    write_json_payload(layout.bundle_summary_path, bundle_summary.model_dump(mode="json"))


def write_payload_bundle_state(
    *,
    layout: PayloadBundleLayout,
    manifest: PayloadBundleManifest,
    inventory: PayloadVisualInventory,
) -> None:
    persist_bundle_models(
        manifest=manifest,
        inventory=inventory,
        manifest_path=layout.manifest_path,
        inventory_path=layout.inventory_path,
    )


def write_debug_render_jobs(*, layout: PayloadBundleLayout, view_entries: list[PayloadViewEntry]) -> None:
    layout.render_jobs_dir.mkdir(parents=True, exist_ok=True)
    for entry in view_entries:
        job_path = layout.render_jobs_dir / f"{entry.view_id}.job.yaml"
        atomic_write_yaml(job_path, build_render_job_payload(entry=entry), sort_keys=False, default_flow_style=False)


__all__ = [
    "remove_stale_payload_bundle_artifacts",
    "write_debug_render_jobs",
    "write_normalized_payload_dump",
    "write_payload_bundle_summary",
    "write_payload_bundle_state",
    "write_payload_bundle_views",
]
