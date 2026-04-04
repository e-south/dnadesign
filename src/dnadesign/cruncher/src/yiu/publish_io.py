"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/publish_io.py

Filesystem write helpers for payload-centric YIU publication.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from dnadesign.cruncher.yiu.bundle_models import PayloadBundleManifest, PayloadViewEntry, PayloadVisualInventory
from dnadesign.cruncher.yiu.publish_layout import PayloadBundleLayout, build_render_job_payload


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
    return path


def write_payload_bundle_views(
    *,
    layout: PayloadBundleLayout,
    payload_contract: dict[str, object],
    split_payload_rows: list[dict[str, object]],
    assembled_payload_contract: dict[str, object],
) -> None:
    _write_json(layout.payload_view_path, payload_contract)
    _write_jsonl(layout.split_payload_view_path, split_payload_rows)
    _write_json(layout.assembled_payload_view_path, assembled_payload_contract)


def write_normalized_payload_dump(
    *,
    layout: PayloadBundleLayout,
    normalized_payload_dump: dict[str, object],
) -> None:
    _write_json(layout.normalized_payload_path, normalized_payload_dump)


def write_payload_bundle_state(
    *,
    layout: PayloadBundleLayout,
    manifest: PayloadBundleManifest,
    inventory: PayloadVisualInventory,
) -> None:
    _write_json(layout.manifest_path, manifest.model_dump(mode="json"))
    _write_json(layout.inventory_path, inventory.model_dump(mode="json"))


def write_debug_render_jobs(*, layout: PayloadBundleLayout, view_entries: list[PayloadViewEntry]) -> None:
    layout.render_jobs_dir.mkdir(parents=True, exist_ok=True)
    for entry in view_entries:
        job_path = layout.render_jobs_dir / f"{entry.view_id}.job.yaml"
        job_path.write_text(yaml.safe_dump(build_render_job_payload(entry=entry), sort_keys=False), encoding="utf-8")


__all__ = [
    "write_debug_render_jobs",
    "write_normalized_payload_dump",
    "write_payload_bundle_state",
    "write_payload_bundle_views",
]
