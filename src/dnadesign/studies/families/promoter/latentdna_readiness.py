"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/families/promoter/latentdna_readiness.py

Read-only LatentDNA readiness inspection for promoter-study status surfaces.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping

from .latentdna_contract import (
    binding_decision_deliverables,
    binding_default_model_family,
    binding_model_families,
    binding_required_wildtypes,
    binding_source_datasets,
    latentdna_binding_path,
    latentdna_doc_path,
    latentdna_workspace_ref,
    load_latentdna_binding,
    load_latentdna_snapshot,
)
from .record_normalizer import PromoterStudyResolvedContext


def inspect_promoter_latentdna_readiness(
    *,
    study_context: PromoterStudyResolvedContext,
) -> dict[str, object]:
    doc_path = latentdna_doc_path(study_context)
    binding_path = latentdna_binding_path(study_context)
    binding_error: str | None = None
    try:
        binding_path, binding = load_latentdna_binding(study_context)
    except ValueError as exc:
        binding = None
        binding_error = str(exc)
    if binding_error is not None:
        return {
            "configured": binding_path is not None and binding_path.is_file(),
            "state": "error",
            "doc": doc_path,
            "binding_ref": None if binding_path is None else str(binding_path),
            "workspace_ref": None,
            "snapshot_ref": None,
            "workspace_id": None,
            "source_datasets": {},
            "decision_deliverables": [],
            "ok_deliverables": [],
            "pending_deliverables": [],
            "export_ids": [],
            "exports_ok": False,
            "browser_default_geometry_ids": [],
            "browser_preferred_hues": [],
            "supported_model_families": [],
            "default_model_family": None,
            "required_wildtype_references": [],
            "last_updated_at": None,
            "error": binding_error,
        }
    if binding is None:
        return {
            "configured": False,
            "state": "not_configured",
            "doc": doc_path,
            "binding_ref": None if binding_path is None else str(binding_path),
            "workspace_ref": None,
            "snapshot_ref": None,
            "workspace_id": None,
            "source_datasets": {},
            "decision_deliverables": [],
            "ok_deliverables": [],
            "pending_deliverables": [],
            "export_ids": [],
            "exports_ok": True,
            "browser_default_geometry_ids": [],
            "browser_preferred_hues": [],
            "supported_model_families": [],
            "default_model_family": None,
            "required_wildtype_references": [],
            "last_updated_at": None,
        }

    workspace_ref = latentdna_workspace_ref(study_context, binding=binding)
    snapshot_error: str | None = None
    try:
        snapshot_path, snapshot = load_latentdna_snapshot(study_context, binding=binding)
    except ValueError as exc:
        snapshot_path = None
        snapshot = None
        snapshot_error = str(exc)
    decision_deliverables = binding_decision_deliverables(binding, snapshot=snapshot)
    source_datasets = binding_source_datasets(binding)
    deliverables_payload = snapshot.get("deliverables") if isinstance(snapshot, Mapping) else {}
    exports_payload = snapshot.get("exports") if isinstance(snapshot, Mapping) else {}
    browser_payload = snapshot.get("browser") if isinstance(snapshot, Mapping) else {}

    ok_deliverables = [
        deliverable_id
        for deliverable_id in decision_deliverables
        if isinstance(deliverables_payload, Mapping)
        and isinstance(deliverables_payload.get(deliverable_id), Mapping)
        and str(deliverables_payload[deliverable_id].get("status") or "") == "ok"
    ]
    pending_deliverables = [
        deliverable_id for deliverable_id in decision_deliverables if deliverable_id not in ok_deliverables
    ]
    export_ids = (
        sorted(str(export_id) for export_id in exports_payload.keys()) if isinstance(exports_payload, Mapping) else []
    )
    export_statuses = [
        str((exports_payload.get(export_id) or {}).get("status") or "")
        for export_id in export_ids
        if isinstance(exports_payload.get(export_id), Mapping)
    ]
    deliverable_statuses = [
        str((deliverables_payload.get(deliverable_id) or {}).get("status") or "")
        for deliverable_id in decision_deliverables
        if isinstance(deliverables_payload.get(deliverable_id), Mapping)
    ]

    if snapshot_error is not None:
        state = "error"
    elif snapshot is None:
        state = "missing"
    elif any(status == "error" for status in [*deliverable_statuses, *export_statuses]):
        state = "error"
    elif pending_deliverables or any(status != "ok" for status in export_statuses):
        state = "attention"
    else:
        state = "ok"

    return {
        "configured": True,
        "state": state,
        "doc": doc_path,
        "binding_ref": None if binding_path is None else str(binding_path),
        "workspace_ref": None if workspace_ref is None else str(workspace_ref),
        "snapshot_ref": None if snapshot_path is None else str(snapshot_path),
        "workspace_id": str(binding.get("workspace_id")),
        "source_datasets": source_datasets,
        "decision_deliverables": decision_deliverables,
        "ok_deliverables": ok_deliverables,
        "pending_deliverables": pending_deliverables,
        "export_ids": export_ids,
        "exports_ok": all(status == "ok" for status in export_statuses) if export_statuses else True,
        "browser_default_geometry_ids": list((browser_payload or {}).get("default_geometry_ids") or []),
        "browser_preferred_hues": list((browser_payload or {}).get("preferred_hues") or []),
        "supported_model_families": binding_model_families(binding)
        or (list(snapshot.get("model_families") or []) if isinstance(snapshot, Mapping) else []),
        "default_model_family": binding_default_model_family(binding),
        "required_wildtype_references": binding_required_wildtypes(binding),
        "last_updated_at": None if snapshot is None else snapshot.get("last_updated_at"),
        **({"error": snapshot_error} if snapshot_error is not None else {}),
    }
