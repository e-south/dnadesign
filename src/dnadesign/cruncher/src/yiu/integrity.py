"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/integrity.py

Bundle integrity checks for YIU v4 publication and `show`.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dnadesign.cruncher.yiu.bundle_models import (
    PayloadBundleManifest,
    PayloadVisualInventory,
    normalized_payload_summary_dump,
    payload_summary_dump,
)
from dnadesign.cruncher.yiu.bundle_paths import (
    resolve_expected_render_artifact_paths,
    resolve_published_plot_path,
)
from dnadesign.cruncher.yiu.bundle_surface import YiuBundleIntegrityState
from dnadesign.cruncher.yiu.domain_models import NormalizedPayload
from dnadesign.cruncher.yiu.errors import YIU_BUNDLE_INVALID, raise_yiu_error
from dnadesign.cruncher.yiu.view_catalog import canonical_payload_view_definitions
from dnadesign.cruncher.yiu.view_io import load_contract_rows
from dnadesign.cruncher.yiu.view_sequence_contracts import (
    build_assembled_payload_view_contract,
    build_split_payload_view_rows,
)


def _fail_bundle(message: str) -> None:
    raise_yiu_error(YIU_BUNDLE_INVALID, message)


def _normalize_view_dump(views: list[object]) -> list[dict[str, object]]:
    return [view.model_dump(mode="json") if hasattr(view, "model_dump") else dict(view) for view in views]


def _validate_payload_summary(*, manifest: PayloadBundleManifest, normalized: NormalizedPayload) -> None:
    expected_summary = normalized_payload_summary_dump(normalized)
    actual_summary = payload_summary_dump(manifest)
    for field_name in expected_summary:
        if actual_summary[field_name] != expected_summary[field_name]:
            _fail_bundle(f"normalized payload and bundle manifest disagree on {field_name}")


def _validate_view_registry(*, manifest: PayloadBundleManifest, inventory: PayloadVisualInventory) -> None:
    expected_view_ids = [entry.view_id for entry in canonical_payload_view_definitions()]
    inventory_view_ids = [view.view_id for view in inventory.views]
    manifest_view_ids = [view.view_id for view in manifest.view_contracts]
    if inventory_view_ids != expected_view_ids:
        _fail_bundle("visual inventory must publish canonical YIU view ids in order: " + ", ".join(expected_view_ids))
    if manifest_view_ids != expected_view_ids:
        _fail_bundle("bundle manifest must publish canonical YIU view ids in order: " + ", ".join(expected_view_ids))


def validate_bundle_state(
    *,
    bundle_dir: Path,
    manifest: PayloadBundleManifest,
    inventory: PayloadVisualInventory,
    normalized: NormalizedPayload,
) -> YiuBundleIntegrityState:
    checks: list[str] = []
    if inventory.view_count != len(inventory.views):
        _fail_bundle("visual inventory view_count does not match the number of published views")
    checks.append("inventory.view_count")
    _validate_view_registry(manifest=manifest, inventory=inventory)
    checks.append("view_registry")
    if manifest.spec_name != inventory.spec_name:
        _fail_bundle("bundle manifest and visual inventory disagree on spec_name")
    checks.append("spec_name")
    if manifest.input_kind != inventory.input_kind:
        _fail_bundle("bundle manifest and visual inventory disagree on input_kind")
    checks.append("input_kind")
    if manifest.render_status != inventory.render_status:
        _fail_bundle("bundle manifest and visual inventory disagree on render_status")
    checks.append("render_status")
    if manifest.composite_render_artifact_path != inventory.composite_render_artifact_path:
        _fail_bundle("bundle manifest and visual inventory disagree on composite render path")
    checks.append("composite_render_path")
    if manifest.published_plot_artifact_path != inventory.published_plot_artifact_path:
        _fail_bundle("bundle manifest and visual inventory disagree on published plot path")
    checks.append("published_plot_path")
    if manifest.pwm_effective != inventory.pwm_effective:
        _fail_bundle("bundle manifest and visual inventory disagree on pwm_effective")
    checks.append("pwm_effective")
    if manifest.payload_view_requires_motif_layers != inventory.payload_view_requires_motif_layers:
        _fail_bundle("bundle manifest and visual inventory disagree on payload_view_requires_motif_layers")
    checks.append("payload_view_requires_motif_layers")
    if _normalize_view_dump(manifest.view_contracts) != _normalize_view_dump(inventory.views):
        _fail_bundle("bundle manifest and visual inventory disagree on published view contracts")
    checks.append("view_contracts")
    _validate_payload_summary(manifest=manifest, normalized=normalized)
    checks.append("summary_fields")
    if manifest.provenance != normalized.source_provenance:
        _fail_bundle("normalized payload and bundle manifest disagree on provenance")
    checks.append("provenance")

    published_rows: dict[str, list[dict[str, Any]]] = {}
    for view in inventory.views:
        contract_path = (bundle_dir / view.view_contract_path).resolve()
        if not contract_path.exists():
            _fail_bundle(f"published view contract is missing: {contract_path}")
        rows = load_contract_rows(contract_path, input_kind=view.input_kind)
        if not rows:
            _fail_bundle(f"published view contract is empty: {contract_path}")
        published_rows[view.view_id] = rows
        actual_kind = str(rows[0].get("contract_kind", "")).strip()
        if actual_kind != view.contract_kind:
            _fail_bundle(f"{view.view_id} contract_kind does not match the manifest entry")
        if view.contract_kind == "yiu_payload_visual_v1":
            actual_version = rows[0].get("schema_version")
            if actual_version != view.schema_version:
                _fail_bundle(f"{view.view_id} schema_version does not match the manifest entry")
    checks.append("published_contracts_present")

    payload_view = published_rows["payload"][0]
    if normalized.motif_context.effective and len(payload_view.get("motif_layers", [])) == 0:
        _fail_bundle("normalized payload says PWM is effective but payload view has zero motif layers")
    if (not inventory.pwm_effective) and payload_view.get("motif_layers"):
        _fail_bundle("inventory claims no PWM while the payload view includes motif layers")
    expected_motif_ids = {motif.motif_instance_id for motif in normalized.motif_context.motifs}
    payload_motif_ids = {motif["motif_instance_id"] for motif in payload_view.get("motif_layers", [])}
    if not payload_motif_ids.issubset(expected_motif_ids):
        _fail_bundle("payload view references motif IDs that are absent from normalized_payload.json")
    if payload_view.get("selected_payload_sequence") != normalized.selected_payload_sequence:
        _fail_bundle("payload view disagrees with normalized selected_payload_sequence")
    if payload_view.get("selected_complement_sequence") != normalized.selected_complement_sequence:
        _fail_bundle("payload view disagrees with normalized selected_complement_sequence")
    if payload_view.get("mismatches") != [entry.model_dump(mode="json") for entry in normalized.mismatches]:
        _fail_bundle("payload view mismatch annotations disagree with normalized_payload.json")
    checks.append("payload_view_consistency")

    expected_assembled = build_assembled_payload_view_contract(normalized)
    if published_rows["assembled_payload"][0] != expected_assembled:
        _fail_bundle("assembled_payload_view.json disagrees with the selected downstream sequences")
    checks.append("assembled_view_consistency")
    expected_split_rows = build_split_payload_view_rows(normalized)
    if published_rows["split_payload"] != expected_split_rows:
        _fail_bundle("split_payload_view.json disagrees with the selected downstream sequences")
    checks.append("split_view_consistency")

    try:
        expected_render_paths = [str(path) for path in resolve_expected_render_artifact_paths(bundle_dir, inventory)]
    except ValueError as exc:
        _fail_bundle(str(exc))
    existing_render_paths = [path for path in expected_render_paths if Path(path).exists()]
    published_plot_path = resolve_published_plot_path(bundle_dir, inventory.published_plot_artifact_path)
    if inventory.published_plot_artifact_path is not None and published_plot_path is None:
        _fail_bundle("bundle inventory declares a published plot path but the workspace root cannot be resolved")
    if inventory.render_status == "rendered":
        missing = [path for path in expected_render_paths if path not in existing_render_paths]
        if missing:
            _fail_bundle("bundle inventory reports rendered outputs that are missing on disk: " + ", ".join(missing))
        if published_plot_path is not None and not published_plot_path.exists():
            _fail_bundle(f"bundle inventory reports a published plot that is missing on disk: {published_plot_path}")
    elif existing_render_paths:
        _fail_bundle(
            "bundle inventory does not report rendered outputs but artifacts exist on disk: "
            + ", ".join(existing_render_paths)
        )
    elif published_plot_path is not None and published_plot_path.exists():
        _fail_bundle(
            "bundle inventory does not report rendered outputs but the published plot exists on disk: "
            + str(published_plot_path)
        )
    checks.append("render_artifacts")
    return YiuBundleIntegrityState(
        checks=checks,
        available_renders=existing_render_paths,
        published_plot_path=None if published_plot_path is None else str(published_plot_path),
        payload_view=payload_view,
        split_rows=published_rows["split_payload"],
    )
