"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/yiu_workflow/show.py

Inspect payload-centric YIU bundles.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.integrity import resolve_outputs_root, validate_bundle_state
from dnadesign.cruncher.yiu.models.bundle import PayloadBundleManifest, PayloadVisualInventory, payload_summary_dump


def _split_row_debug(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    debug_rows: list[dict[str, object]] = []
    for row in rows:
        meta = row.get("meta", {})
        if not isinstance(meta, dict):
            continue
        debug_rows.append(
            {
                "fragment_side": meta.get("fragment_side"),
                "panel_order": meta.get("panel_order"),
                "selected_sticky_end_sequence_5to3": meta.get("selected_sticky_end_sequence_5to3"),
                "canonical_sticky_end_sequence_5to3": meta.get("canonical_sticky_end_sequence_5to3"),
                "retained_primary_sequence_5to3": meta.get("retained_primary_sequence_5to3"),
                "retained_complement_sequence_3to5": meta.get("retained_complement_sequence_3to5"),
                "retained_payload_body_sequence_5to3": meta.get("retained_payload_body_sequence_5to3"),
                "sticky_end_display_span": meta.get("sticky_end_display_span"),
                "payload_body_display_span": meta.get("payload_body_display_span"),
                "payload_junction_window": meta.get("payload_junction_window"),
                "ghost_excised_context": meta.get("ghost_excised_context"),
            }
        )
    return debug_rows


def show_yiu_bundle(bundle_dir: str | Path, *, verbose: bool = False) -> dict[str, object]:
    resolved = Path(bundle_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"YIU bundle directory not found: {resolved}")
    manifest_path = resolved / "bundle_manifest.json"
    normalized_path = resolved / "normalized_payload.json"
    inventory_path = resolved / "visual_inventory.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"YIU bundle manifest not found: {manifest_path}")
    if not normalized_path.exists():
        raise FileNotFoundError(f"YIU normalized payload not found: {normalized_path}")
    if not inventory_path.exists():
        raise FileNotFoundError(f"YIU visual inventory not found: {inventory_path}")

    manifest = PayloadBundleManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    normalized = NormalizedPayload.model_validate(json.loads(normalized_path.read_text(encoding="utf-8")))
    inventory = PayloadVisualInventory.model_validate(json.loads(inventory_path.read_text(encoding="utf-8")))
    integrity = validate_bundle_state(
        bundle_dir=resolved, manifest=manifest, inventory=inventory, normalized=normalized
    )
    outputs_root = resolve_outputs_root(resolved)
    payload = {
        "bundle_dir": str(resolved),
        "outputs_root": None if outputs_root is None else str(outputs_root.resolve()),
        "bundle_contract": manifest.bundle_contract,
        "provenance": manifest.provenance,
        **payload_summary_dump(manifest),
        "view_ids": [view.view_id for view in inventory.views],
        "render_status": inventory.render_status,
        "available_renders": integrity["available_renders"],
        "composite_render_artifact_path": None
        if inventory.composite_render_artifact_path is None
        else str((resolved / inventory.composite_render_artifact_path).resolve()),
        "published_plot_artifact_path": integrity["published_plot_path"],
        "bundle_manifest_path": str(manifest_path.resolve()),
        "normalized_payload_path": str(normalized_path.resolve()),
        "visual_inventory_path": str(inventory_path.resolve()),
        "integrity": {
            "status": "ok",
            "checks": integrity["checks"],
            "available_render_count": len(integrity["available_renders"]),
        },
        "optimization_decision": normalized.optimization_decision.model_dump(mode="json"),
        "motif_context": normalized.motif_context.model_dump(mode="json"),
    }
    if verbose:
        payload["split_row_debug"] = _split_row_debug(integrity["split_rows"])
    return payload
