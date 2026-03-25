"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/record_loader.py

Loads the OPS-facing checked-in study contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import StudyOpsContract


def load_study_ops_contract(study_root: Path) -> StudyOpsContract:
    contract_path = study_root / "ops.study.yaml"
    if not contract_path.exists():
        raise ValueError(f"study record missing ops.study.yaml: {contract_path}")
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ops.study.yaml must be a mapping: {contract_path}")
    version = int(payload.get("version") or 0)
    if version != 1:
        raise ValueError(f"unsupported ops.study.yaml version {version}: {contract_path}")

    study_id = str(payload.get("study_id") or "").strip()
    family = str(payload.get("family") or "").strip()
    if not study_id:
        raise ValueError(f"ops.study.yaml must define study_id: {contract_path}")
    if not family:
        raise ValueError(f"ops.study.yaml must define family: {contract_path}")

    phase_order_payload = payload.get("phase_order") or []
    if not isinstance(phase_order_payload, list) or not phase_order_payload:
        raise ValueError(f"ops.study.yaml must define a non-empty phase_order list: {contract_path}")
    phase_order = tuple(str(item).strip() for item in phase_order_payload if str(item).strip())
    if not phase_order:
        raise ValueError(f"ops.study.yaml phase_order must contain non-empty phase ids: {contract_path}")

    snapshot_payload = payload.get("snapshot") or {}
    if snapshot_payload and not isinstance(snapshot_payload, dict):
        raise ValueError(f"ops.study.yaml snapshot must be a mapping: {contract_path}")
    preflight_payload = payload.get("preflight") or {}
    if not isinstance(preflight_payload, dict):
        raise ValueError(f"ops.study.yaml preflight must be a mapping: {contract_path}")
    next_scope_payload = preflight_payload.get("next_scope") or {}
    if next_scope_payload and not isinstance(next_scope_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.next_scope must be a mapping: {contract_path}")
    phase_groups_payload = next_scope_payload.get("phase_groups") or {}
    if phase_groups_payload and not isinstance(phase_groups_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.next_scope.phase_groups must be a mapping: {contract_path}")
    phase_targets_payload = preflight_payload.get("phase_targets") or {}
    if phase_targets_payload and not isinstance(phase_targets_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.phase_targets must be a mapping: {contract_path}")

    phase_groups: dict[str, tuple[str, ...]] = {}
    for phase_id, groups_payload in phase_groups_payload.items():
        groups = tuple(str(item).strip() for item in groups_payload or [] if str(item).strip())
        phase_groups[str(phase_id).strip()] = groups

    phase_targets = {
        str(name).strip(): str(phase_id).strip()
        for name, phase_id in phase_targets_payload.items()
        if str(name).strip() and str(phase_id).strip()
    }
    infer_lane_groups = tuple(
        str(item).strip() for item in next_scope_payload.get("infer_lane_groups") or () if str(item).strip()
    )

    return StudyOpsContract(
        study_id=study_id,
        family=family,
        phase_order=phase_order,
        snapshot_summary_scope=str(snapshot_payload.get("summary_scope") or "repo").strip() or "repo",
        preflight_default_scope=str(preflight_payload.get("default_scope") or "next").strip() or "next",
        preflight_phase_targets=phase_targets,
        next_scope_phase_groups=phase_groups,
        infer_lane_groups=infer_lane_groups or ("infer", "notify", "infer_batch_plan"),
        raw_payload=dict(payload),
    )


__all__ = ["load_study_ops_contract"]
