"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/bundle_state.py

Shared bundle-state load/persist helpers for YIU publication, rendering, and
inspection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeVar

from dnadesign.cruncher.yiu.bundle_models import PayloadBundleManifest, PayloadViewEntry, PayloadVisualInventory
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.view_io import write_json_payload

RenderStatus = Literal["not_requested", "rendered", "missing", "partial", "failed"]
BundleModelT = TypeVar("BundleModelT", PayloadBundleManifest, PayloadVisualInventory, NormalizedPayload)


def _load_bundle_model(path: Path, *, model_type: type[BundleModelT], label: str) -> BundleModelT:
    if not path.exists():
        raise FileNotFoundError(f"YIU {label} not found: {path}")
    return model_type.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class YiuBundleStatePaths:
    bundle_dir: Path
    bundle_summary_path: Path
    manifest_path: Path
    normalized_payload_path: Path
    inventory_path: Path


@dataclass(frozen=True)
class YiuBundleState:
    paths: YiuBundleStatePaths
    manifest: PayloadBundleManifest
    inventory: PayloadVisualInventory
    normalized: NormalizedPayload | None = None

    @property
    def bundle_dir(self) -> Path:
        return self.paths.bundle_dir

    def persist(self) -> None:
        persist_bundle_models(
            manifest=self.manifest,
            inventory=self.inventory,
            manifest_path=self.paths.manifest_path,
            inventory_path=self.paths.inventory_path,
        )

    def with_render_state(
        self,
        *,
        rendered_count: int,
        last_rendered_at: str | None,
        views: list[PayloadViewEntry],
        render_status: RenderStatus,
    ) -> YiuBundleState:
        updated_inventory = self.inventory.model_copy(
            update={
                "render_count": rendered_count,
                "render_status": render_status,
                "last_rendered_at": last_rendered_at,
                "views": views,
            }
        )
        updated_manifest = self.manifest.model_copy(
            update={
                "render_status": updated_inventory.render_status,
                "view_contracts": updated_inventory.views,
            }
        )
        return replace(self, manifest=updated_manifest, inventory=updated_inventory)


def resolve_bundle_state_paths(bundle_dir: str | Path) -> YiuBundleStatePaths:
    resolved = Path(bundle_dir).expanduser().resolve()
    return YiuBundleStatePaths(
        bundle_dir=resolved,
        bundle_summary_path=resolved / "bundle_summary.json",
        manifest_path=resolved / "bundle_manifest.json",
        normalized_payload_path=resolved / "normalized_payload.json",
        inventory_path=resolved / "visual_inventory.json",
    )


def persist_bundle_models(
    *,
    manifest: PayloadBundleManifest,
    inventory: PayloadVisualInventory,
    manifest_path: Path,
    inventory_path: Path,
) -> None:
    write_json_payload(manifest_path, manifest.model_dump(mode="json"))
    write_json_payload(inventory_path, inventory.model_dump(mode="json"))


def load_bundle_state(bundle_dir: str | Path, *, include_normalized: bool = False) -> YiuBundleState:
    paths = resolve_bundle_state_paths(bundle_dir)
    manifest = _load_bundle_model(paths.manifest_path, model_type=PayloadBundleManifest, label="bundle manifest")
    inventory = _load_bundle_model(paths.inventory_path, model_type=PayloadVisualInventory, label="visual inventory")
    normalized = None
    if include_normalized:
        normalized = _load_bundle_model(
            paths.normalized_payload_path,
            model_type=NormalizedPayload,
            label="normalized payload",
        )
    return YiuBundleState(
        paths=paths,
        manifest=manifest,
        inventory=inventory,
        normalized=normalized,
    )


__all__ = [
    "RenderStatus",
    "YiuBundleState",
    "YiuBundleStatePaths",
    "load_bundle_state",
    "persist_bundle_models",
    "resolve_bundle_state_paths",
]
