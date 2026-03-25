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

from dnadesign.ops.status.path_ref import resolve_path_ref

from .models import (
    STUDY_PHASE_STATUSES,
    STUDY_PREFLIGHT_SCOPES,
    STUDY_SUMMARY_SCOPES,
    StudyOpsContract,
    StudyPhaseContract,
    StudyPreflightContract,
    StudyPreflightNextScopeContract,
)


def load_study_ops_contract(study_root: Path) -> StudyOpsContract:
    resolved_study_root = study_root.expanduser().resolve()
    repo_root = _discover_repo_root(resolved_study_root)
    contract_path = resolved_study_root / "ops.study.yaml"
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
    phase_order = _string_sequence(
        phase_order_payload,
        label="ops.study.yaml phase_order",
        source=contract_path,
    )

    current_phase_payload = payload.get("current_phase") or {}
    if current_phase_payload and not isinstance(current_phase_payload, dict):
        raise ValueError(f"ops.study.yaml current_phase must be a mapping: {contract_path}")
    current_phase_strategy = str(current_phase_payload.get("strategy") or "explicit").strip().lower()
    if current_phase_strategy not in {"explicit", "derive_from_phase_status"}:
        raise ValueError(
            f"ops.study.yaml current_phase.strategy must be one of: explicit, derive_from_phase_status: {contract_path}"
        )
    current_phase_id = str(current_phase_payload.get("id") or "").strip() or None

    phases_payload = payload.get("phases") or []
    if not isinstance(phases_payload, list) or not phases_payload:
        raise ValueError(f"ops.study.yaml must define a non-empty phases list: {contract_path}")
    phases: list[StudyPhaseContract] = []
    seen_phase_ids: set[str] = set()
    for index, item in enumerate(phases_payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"ops.study.yaml phase entry {index} must be a mapping: {contract_path}")
        phase_id = str(item.get("id") or "").strip()
        phase_status = str(item.get("status") or "").strip()
        if not phase_id:
            raise ValueError(f"ops.study.yaml phase entry {index} must define id: {contract_path}")
        if not phase_status:
            raise ValueError(f"ops.study.yaml phase {phase_id} must define status: {contract_path}")
        if phase_status not in STUDY_PHASE_STATUSES:
            allowed_statuses = ", ".join(sorted(STUDY_PHASE_STATUSES))
            raise ValueError(
                f"ops.study.yaml phase {phase_id} has unsupported status {phase_status!r}; "
                f"expected one of: {allowed_statuses}: {contract_path}"
            )
        if phase_id in seen_phase_ids:
            raise ValueError(f"ops.study.yaml phases must not duplicate id {phase_id!r}: {contract_path}")
        seen_phase_ids.add(phase_id)
        phases.append(
            StudyPhaseContract(
                id=phase_id,
                status=phase_status,
                next_surface=_validated_surface_ref(
                    item.get("next_surface"),
                    repo_root=repo_root,
                    study_root=resolved_study_root,
                    label=f"ops.study.yaml phase {phase_id} next_surface",
                ),
                blocker=_string_or_none(item.get("blocker")),
                output_dataset=_string_or_none(item.get("output_dataset")),
                primary_dataset=_string_or_none(item.get("primary_dataset")),
            )
        )

    phase_ids = tuple(phase.id for phase in phases)
    if phase_order != phase_ids:
        raise ValueError(f"ops.study.yaml phase_order must match phases ids in the same order: {contract_path}")
    if current_phase_strategy == "explicit":
        if current_phase_id is None:
            raise ValueError(f"ops.study.yaml current_phase.id must be defined for explicit strategy: {contract_path}")
        if current_phase_id not in seen_phase_ids:
            raise ValueError(
                f"ops.study.yaml current_phase.id {current_phase_id!r} is not declared under phases: {contract_path}"
            )
    else:
        current_phase_id = _derive_current_phase_id(phases)

    snapshot_payload = payload.get("snapshot") or {}
    if snapshot_payload and not isinstance(snapshot_payload, dict):
        raise ValueError(f"ops.study.yaml snapshot must be a mapping: {contract_path}")
    preflight_payload = payload.get("preflight") or {}
    if not isinstance(preflight_payload, dict):
        raise ValueError(f"ops.study.yaml preflight must be a mapping: {contract_path}")
    default_scope = str(preflight_payload.get("default_scope") or "").strip()
    if not default_scope:
        raise ValueError(f"ops.study.yaml preflight.default_scope must be non-empty: {contract_path}")
    if default_scope not in STUDY_PREFLIGHT_SCOPES:
        allowed_scopes = ", ".join(sorted(STUDY_PREFLIGHT_SCOPES))
        raise ValueError(f"ops.study.yaml preflight.default_scope must be one of: {allowed_scopes}: {contract_path}")
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
        normalized_phase_id = str(phase_id or "").strip()
        if not normalized_phase_id:
            raise ValueError(
                f"ops.study.yaml preflight.next_scope.target_phase_groups keys must be non-empty: {contract_path}"
            )
        if normalized_phase_id not in seen_phase_ids:
            raise ValueError(
                f"ops.study.yaml preflight.next_scope.target_phase_groups references undeclared phase "
                f"{normalized_phase_id!r}: {contract_path}"
            )
        target_phase_groups[normalized_phase_id] = _string_sequence(
            groups_payload or [],
            label=f"ops.study.yaml preflight.next_scope.target_phase_groups.{normalized_phase_id}",
            source=contract_path,
            allow_empty=True,
        )

    group_phase_bindings: dict[str, str] = {}
    for raw_group, raw_phase_id in group_phase_bindings_payload.items():
        group = str(raw_group or "").strip()
        phase_id = str(raw_phase_id or "").strip()
        if not group:
            raise ValueError(f"ops.study.yaml preflight.group_phase_bindings keys must be non-empty: {contract_path}")
        if not phase_id:
            raise ValueError(
                f"ops.study.yaml preflight.group_phase_bindings.{group} must be a non-empty phase id: {contract_path}"
            )
        if phase_id not in seen_phase_ids:
            raise ValueError(
                f"ops.study.yaml preflight.group_phase_bindings.{group} references undeclared phase "
                f"{phase_id!r}: {contract_path}"
            )
        group_phase_bindings[group] = phase_id

    runtime_phase_groups = _string_sequence(
        next_scope_payload.get("runtime_phase_groups") or [],
        label="ops.study.yaml preflight.next_scope.runtime_phase_groups",
        source=contract_path,
        allow_empty=True,
    )
    runtime_shared_groups = _string_sequence(
        next_scope_payload.get("runtime_shared_groups") or [],
        label="ops.study.yaml preflight.next_scope.runtime_shared_groups",
        source=contract_path,
        allow_empty=True,
    )
    summary_scope = str(snapshot_payload.get("summary_scope") or "repo").strip()
    if not summary_scope:
        raise ValueError(f"ops.study.yaml snapshot.summary_scope must be non-empty: {contract_path}")
    if summary_scope not in STUDY_SUMMARY_SCOPES:
        allowed_scopes = ", ".join(sorted(STUDY_SUMMARY_SCOPES))
        raise ValueError(f"ops.study.yaml snapshot.summary_scope must be one of: {allowed_scopes}: {contract_path}")

    return StudyOpsContract(
        study_id=study_id,
        family=family,
        phase_order=phase_order,
        current_phase_id=current_phase_id,
        phases=tuple(phases),
        snapshot_summary_scope=summary_scope,
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


def _derive_current_phase_id(phases: list[StudyPhaseContract]) -> str | None:
    ordered_statuses = ("in_progress", "ready", "planned", "blocked_gpu", "blocked", "parallel_optional")
    for status in ordered_statuses:
        for phase in phases:
            if phase.status == status:
                return phase.id
    return None


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_sequence(
    values: object,
    *,
    label: str,
    source: Path,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list: {source}")
    items: list[str] = []
    seen: set[str] = set()
    for index, raw_value in enumerate(values, start=1):
        text = str(raw_value or "").strip()
        if not text:
            raise ValueError(f"{label} entry {index} must be non-empty: {source}")
        if text in seen:
            raise ValueError(f"{label} must not duplicate {text!r}: {source}")
        seen.add(text)
        items.append(text)
    if not items and not allow_empty:
        raise ValueError(f"{label} must not be empty: {source}")
    return tuple(items)


def _validated_surface_ref(
    value: object,
    *,
    repo_root: Path,
    study_root: Path,
    label: str,
) -> str | None:
    text = _string_or_none(value)
    if text is None:
        return None
    resolve_path_ref(
        text,
        repo_root=repo_root,
        manifest_dir=study_root,
        default_base="repo",
        label=label,
    )
    return text


def _discover_repo_root(study_root: Path) -> Path:
    for parent in (study_root, *study_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise ValueError(f"study record must live inside a dnadesign repository checkout: {study_root}")


__all__ = ["load_study_ops_contract"]
