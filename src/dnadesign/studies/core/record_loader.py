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

from .models import StudyOpsContract, StudyPreflightContract, StudyPreflightNextScopeContract


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
    default_scope = str(preflight_payload.get("default_scope") or "").strip()
    if not default_scope:
        raise ValueError(f"ops.study.yaml preflight.default_scope must be non-empty: {contract_path}")
    if "group_phase_bindings" not in preflight_payload:
        raise ValueError(f"ops.study.yaml preflight.group_phase_bindings must be defined: {contract_path}")
    if "next_scope" not in preflight_payload:
        raise ValueError(f"ops.study.yaml preflight.next_scope must be defined: {contract_path}")
    next_scope_payload = preflight_payload.get("next_scope") or {}
    if next_scope_payload and not isinstance(next_scope_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.next_scope must be a mapping: {contract_path}")
    if "target_phase_groups" not in next_scope_payload:
        raise ValueError(f"ops.study.yaml preflight.next_scope.target_phase_groups must be defined: {contract_path}")
    if "runtime_phase_groups" not in next_scope_payload:
        raise ValueError(f"ops.study.yaml preflight.next_scope.runtime_phase_groups must be defined: {contract_path}")
    if "runtime_shared_groups" not in next_scope_payload:
        raise ValueError(f"ops.study.yaml preflight.next_scope.runtime_shared_groups must be defined: {contract_path}")
    target_phase_groups_payload = next_scope_payload.get("target_phase_groups") or {}
    if target_phase_groups_payload and not isinstance(target_phase_groups_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.next_scope.target_phase_groups must be a mapping: {contract_path}")
    group_phase_bindings_payload = preflight_payload.get("group_phase_bindings") or {}
    if group_phase_bindings_payload and not isinstance(group_phase_bindings_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.group_phase_bindings must be a mapping: {contract_path}")

    target_phase_groups: dict[str, tuple[str, ...]] = {}
    for phase_id, groups_payload in target_phase_groups_payload.items():
        groups = tuple(str(item).strip() for item in groups_payload or [] if str(item).strip())
        target_phase_groups[str(phase_id).strip()] = groups

    group_phase_bindings = {
        str(name).strip(): str(phase_id).strip()
        for name, phase_id in group_phase_bindings_payload.items()
        if str(name).strip() and str(phase_id).strip()
    }
    runtime_phase_groups = tuple(
        str(item).strip() for item in next_scope_payload.get("runtime_phase_groups") or () if str(item).strip()
    )
    runtime_shared_groups = tuple(
        str(item).strip() for item in next_scope_payload.get("runtime_shared_groups") or () if str(item).strip()
    )

    return StudyOpsContract(
        study_id=study_id,
        family=family,
        phase_order=phase_order,
        snapshot_summary_scope=str(snapshot_payload.get("summary_scope") or "repo").strip() or "repo",
        preflight=StudyPreflightContract(
            default_scope=default_scope,
            group_phase_bindings=group_phase_bindings,
            next_scope=StudyPreflightNextScopeContract(
                target_phase_groups=target_phase_groups,
                runtime_phase_groups=runtime_phase_groups,
                runtime_shared_groups=runtime_shared_groups,
            ),
        ),
        raw_payload=dict(payload),
    )


__all__ = ["load_study_ops_contract"]
