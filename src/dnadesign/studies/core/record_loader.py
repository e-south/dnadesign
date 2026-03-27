"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/core/record_loader.py

Loads the OPS-facing checked-in study contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from dnadesign.ops.preflight.models import supported_preflight_check_kinds
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
    if version != 2:
        raise ValueError(f"unsupported ops.study.yaml version {version}: {contract_path}")

    study_id = str(payload.get("study_id") or "").strip()
    family = str(payload.get("family") or "").strip()
    title = _string_or_none(payload.get("title"))
    if not study_id:
        raise ValueError(f"ops.study.yaml must define study_id: {contract_path}")
    if not family:
        raise ValueError(f"ops.study.yaml must define family: {contract_path}")

    record_sources = _validated_contract_refs_mapping(
        payload.get("record_sources"),
        repo_root=repo_root,
        study_root=resolved_study_root,
        contract_path=contract_path,
        label="ops.study.yaml record_sources",
    )

    lifecycle_payload = payload.get("lifecycle") or {}
    if not isinstance(lifecycle_payload, dict):
        raise ValueError(f"ops.study.yaml lifecycle must be a mapping: {contract_path}")
    phase_order_payload = lifecycle_payload.get("phase_order") or []
    if not isinstance(phase_order_payload, list) or not phase_order_payload:
        raise ValueError(f"ops.study.yaml lifecycle.phase_order must define a non-empty list: {contract_path}")
    phase_order = _string_sequence(
        phase_order_payload,
        label="ops.study.yaml lifecycle.phase_order",
        source=contract_path,
    )

    current_phase_payload = lifecycle_payload.get("current_phase") or {}
    if current_phase_payload and not isinstance(current_phase_payload, dict):
        raise ValueError(f"ops.study.yaml lifecycle.current_phase must be a mapping: {contract_path}")
    current_phase_strategy = str(current_phase_payload.get("strategy") or "explicit").strip().lower()
    if current_phase_strategy not in {"explicit", "derive_from_phase_status"}:
        raise ValueError(
            "ops.study.yaml lifecycle.current_phase.strategy must be one of: "
            f"explicit, derive_from_phase_status: {contract_path}"
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
        raise ValueError(
            f"ops.study.yaml lifecycle.phase_order must match phases ids in the same order: {contract_path}"
        )
    if current_phase_strategy == "explicit":
        if current_phase_id is None:
            raise ValueError(
                f"ops.study.yaml lifecycle.current_phase.id must be defined for explicit strategy: {contract_path}"
            )
        if current_phase_id not in seen_phase_ids:
            raise ValueError(
                "ops.study.yaml lifecycle.current_phase.id "
                f"{current_phase_id!r} is not declared under phases: {contract_path}"
            )
    else:
        current_phase_id = _derive_current_phase_id(phases)

    artifacts = _validated_contract_named_payloads(
        payload.get("artifacts"),
        repo_root=repo_root,
        study_root=resolved_study_root,
        contract_path=contract_path,
        label="ops.study.yaml artifacts",
    )
    execution_surfaces = _validated_execution_surfaces(
        payload.get("execution_surfaces"),
        repo_root=repo_root,
        study_root=resolved_study_root,
        contract_path=contract_path,
        label="ops.study.yaml execution_surfaces",
    )

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
    scopes_payload = preflight_payload.get("scopes") or {}
    if scopes_payload and not isinstance(scopes_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.scopes must be a mapping: {contract_path}")
    if "group_phase_bindings" not in preflight_payload:
        raise ValueError(f"ops.study.yaml preflight.group_phase_bindings must be defined: {contract_path}")
    if "next_scope" not in preflight_payload:
        raise ValueError(f"ops.study.yaml preflight.next_scope must be defined: {contract_path}")
    checks_payload = preflight_payload.get("checks") or {}
    if checks_payload and not isinstance(checks_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.checks must be a mapping: {contract_path}")
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
    known_preflight_groups = {
        *group_phase_bindings,
        *(group for groups in target_phase_groups.values() for group in groups),
        *runtime_phase_groups,
        *runtime_shared_groups,
    }
    scope_payloads = _validated_preflight_scopes(scopes_payload, contract_path=contract_path)
    check_specs = _validated_preflight_checks(
        checks_payload,
        phase_ids=seen_phase_ids,
        known_groups=known_preflight_groups,
        artifact_ids=set(artifacts),
        execution_surface_ids=set(execution_surfaces),
        execution_surfaces=execution_surfaces,
        contract_path=contract_path,
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
        title=title,
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
            scope_payloads=scope_payloads,
            check_specs=check_specs,
        ),
        record_sources=record_sources,
        artifacts=artifacts,
        execution_surfaces=execution_surfaces,
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
    allow_duplicates: bool = False,
) -> tuple[str, ...]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list: {source}")
    items: list[str] = []
    seen: set[str] = set()
    for index, raw_value in enumerate(values, start=1):
        text = str(raw_value or "").strip()
        if not text:
            raise ValueError(f"{label} entry {index} must be non-empty: {source}")
        if text in seen and not allow_duplicates:
            raise ValueError(f"{label} must not duplicate {text!r}: {source}")
        seen.add(text)
        items.append(text)
    if not items and not allow_empty:
        raise ValueError(f"{label} must not be empty: {source}")
    return tuple(items)


def _validated_contract_refs_mapping(
    value: object,
    *,
    repo_root: Path,
    study_root: Path,
    contract_path: Path,
    label: str,
) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping: {contract_path}")
    resolved: dict[str, str] = {}
    for raw_key, raw_value in value.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError(f"{label} keys must be non-empty: {contract_path}")
        ref = _string_or_none(raw_value)
        if ref is None:
            raise ValueError(f"{label}.{key} must be a non-empty path ref: {contract_path}")
        _validate_contract_path_ref(
            ref,
            repo_root=repo_root,
            study_root=study_root,
            label=f"{label}.{key}",
        )
        resolved[key] = ref
    return resolved


def _validated_contract_named_payloads(
    value: object,
    *,
    repo_root: Path,
    study_root: Path,
    contract_path: Path,
    label: str,
) -> dict[str, dict[str, object]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping: {contract_path}")
    resolved: dict[str, dict[str, object]] = {}
    for raw_key, raw_payload in value.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError(f"{label} keys must be non-empty: {contract_path}")
        if not isinstance(raw_payload, dict):
            raise ValueError(f"{label}.{key} must be a mapping: {contract_path}")
        normalized_payload = dict(raw_payload)
        for field_name, field_value in normalized_payload.items():
            normalized_field = str(field_name or "").strip()
            if normalized_field.endswith("_ref") or normalized_field == "ref":
                ref = _string_or_none(field_value)
                if ref is None:
                    raise ValueError(f"{label}.{key}.{normalized_field} must be a non-empty path ref: {contract_path}")
                _validate_contract_path_ref(
                    ref,
                    repo_root=repo_root,
                    study_root=study_root,
                    label=f"{label}.{key}.{normalized_field}",
                )
        resolved[key] = normalized_payload
    return resolved


def _validated_execution_surfaces(
    value: object,
    *,
    repo_root: Path,
    study_root: Path,
    contract_path: Path,
    label: str,
) -> dict[str, dict[str, object]]:
    resolved = _validated_contract_named_payloads(
        value,
        repo_root=repo_root,
        study_root=study_root,
        contract_path=contract_path,
        label=label,
    )
    for surface_id, payload in resolved.items():
        surface_type = _string_or_none(payload.get("surface_type"))
        if surface_type is None:
            raise ValueError(f"{label}.{surface_id} must define surface_type: {contract_path}")
        if surface_type == "runbook":
            if _string_or_none(payload.get("runbook_ref")) is None:
                raise ValueError(f"{label}.{surface_id} must define runbook_ref: {contract_path}")
        elif surface_type == "workspace":
            if _string_or_none(payload.get("workspace_ref")) is None:
                raise ValueError(f"{label}.{surface_id} must define workspace_ref: {contract_path}")
        elif surface_type == "command":
            argv = _string_sequence(
                payload.get("argv"),
                label=f"{label}.{surface_id} argv",
                source=contract_path,
                allow_duplicates=True,
            )
            payload["argv"] = list(argv)
            cwd_ref = payload.get("cwd_ref")
            if cwd_ref is not None:
                normalized_cwd_ref = _string_or_none(cwd_ref)
                if normalized_cwd_ref is None:
                    raise ValueError(f"{label}.{surface_id}.cwd_ref must be a non-empty path ref: {contract_path}")
                _validate_contract_path_ref(
                    normalized_cwd_ref,
                    repo_root=repo_root,
                    study_root=study_root,
                    label=f"{label}.{surface_id}.cwd_ref",
                )
        elif surface_type == "scheduler":
            backend = _string_or_none(payload.get("backend"))
            if backend is None:
                raise ValueError(f"{label}.{surface_id} must define backend: {contract_path}")
        else:
            raise ValueError(f"{label}.{surface_id} has unsupported surface_type {surface_type!r}: {contract_path}")
    return resolved


def _validated_preflight_scopes(
    scopes_payload: object,
    *,
    contract_path: Path,
) -> dict[str, dict[str, object]]:
    if scopes_payload is None:
        return {}
    if not isinstance(scopes_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.scopes must be a mapping: {contract_path}")
    normalized: dict[str, dict[str, object]] = {}
    for raw_scope, raw_payload in scopes_payload.items():
        scope = str(raw_scope or "").strip()
        if not scope:
            raise ValueError(f"ops.study.yaml preflight.scopes keys must be non-empty: {contract_path}")
        if scope not in STUDY_PREFLIGHT_SCOPES:
            allowed_scopes = ", ".join(sorted(STUDY_PREFLIGHT_SCOPES))
            raise ValueError(
                f"ops.study.yaml preflight.scopes.{scope} must use one of: {allowed_scopes}: {contract_path}"
            )
        if not isinstance(raw_payload, dict):
            raise ValueError(f"ops.study.yaml preflight.scopes.{scope} must be a mapping: {contract_path}")
        normalized[scope] = dict(raw_payload)
    return normalized


def _validated_preflight_checks(
    checks_payload: object,
    *,
    phase_ids: set[str],
    known_groups: set[str],
    artifact_ids: set[str],
    execution_surface_ids: set[str],
    execution_surfaces: Mapping[str, Mapping[str, object]],
    contract_path: Path,
) -> dict[str, tuple[dict[str, object], ...]]:
    if checks_payload is None:
        return {}
    if not isinstance(checks_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.checks must be a mapping: {contract_path}")
    resolved: dict[str, tuple[dict[str, object], ...]] = {}
    seen_check_ids: set[str] = set()
    for raw_phase_id, raw_specs in checks_payload.items():
        phase_id = str(raw_phase_id or "").strip()
        if not phase_id:
            raise ValueError(f"ops.study.yaml preflight.checks keys must be non-empty: {contract_path}")
        if phase_id not in phase_ids:
            raise ValueError(
                f"ops.study.yaml preflight.checks references undeclared phase {phase_id!r}: {contract_path}"
            )
        if not isinstance(raw_specs, list):
            raise ValueError(f"ops.study.yaml preflight.checks.{phase_id} must be a list: {contract_path}")
        specs: list[dict[str, object]] = []
        for index, raw_spec in enumerate(raw_specs, start=1):
            if not isinstance(raw_spec, dict):
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {index} must be a mapping: {contract_path}"
                )
            spec = dict(raw_spec)
            kind = _string_or_none(spec.get("kind"))
            if kind is None:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {index} must define kind: {contract_path}"
                )
            supported_kinds = supported_preflight_check_kinds()
            if kind not in supported_kinds:
                allowed_kinds = ", ".join(sorted(supported_kinds))
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {index} has unsupported kind {kind!r}; "
                    f"expected one of: {allowed_kinds}: {contract_path}"
                )
            check_id = _string_or_none(spec.get("check_id"))
            if check_id is None:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {index} must define check_id: {contract_path}"
                )
            if check_id in seen_check_ids:
                raise ValueError(
                    f"ops.study.yaml preflight.checks must not duplicate check_id {check_id!r}: {contract_path}"
                )
            seen_check_ids.add(check_id)
            check_group = _string_or_none(spec.get("check_group"))
            if check_group is None:
                raise ValueError(
                    "ops.study.yaml preflight.checks."
                    f"{phase_id} entry {check_id} must define check_group: {contract_path}"
                )
            if check_group not in known_groups:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} references unknown check_group "
                    f"{check_group!r}: {contract_path}"
                )
            summary = _string_or_none(spec.get("summary"))
            if summary is None:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} must define summary: {contract_path}"
                )
            explicit_phase_id = _string_or_none(spec.get("phase_id"))
            if explicit_phase_id is not None and explicit_phase_id not in phase_ids:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} references undeclared phase_id "
                    f"{explicit_phase_id!r}: {contract_path}"
                )
            required = spec.get("required")
            if required is not None and not isinstance(required, bool):
                raise ValueError(
                    "ops.study.yaml preflight.checks."
                    f"{phase_id} entry {check_id} must use boolean required: {contract_path}"
                )

            if kind in {"path_exists", "dataset_snapshot"}:
                artifact = _string_or_none(spec.get("artifact"))
                if artifact is None:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} must define artifact: {contract_path}"
                    )
                if artifact not in artifact_ids:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} references unknown artifact "
                        f"{artifact!r}: {contract_path}"
                    )
            if kind == "dataset_snapshot":
                target_rows = spec.get("target_rows")
                if not isinstance(target_rows, int) or target_rows <= 0:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} must define positive integer "
                        f"target_rows: {contract_path}"
                    )
            if kind in {"runbook_plan", "workspace_layout", "command", "scheduler_queue"}:
                surface = _string_or_none(spec.get("surface"))
                if surface is None:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} must define surface: {contract_path}"
                    )
                if surface not in execution_surface_ids:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} references unknown surface "
                        f"{surface!r}: {contract_path}"
                    )
                surface_type = str((execution_surfaces.get(surface) or {}).get("surface_type") or "").strip()
                expected_surface_type = {
                    "runbook_plan": "runbook",
                    "workspace_layout": "workspace",
                    "command": "command",
                    "scheduler_queue": "scheduler",
                }[kind]
                if surface_type != expected_surface_type:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} requires surface {surface!r} to use "
                        f"surface_type {expected_surface_type!r}: {contract_path}"
                    )
            if kind == "environment":
                vars_payload = spec.get("vars")
                vars_list = _string_sequence(
                    vars_payload,
                    label=f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} vars",
                    source=contract_path,
                )
                spec["vars"] = list(vars_list)
                match_mode = _string_or_none(spec.get("match_mode")) or "all"
                if match_mode not in {"all", "any"}:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} match_mode must be one of: all, any: {contract_path}"
                    )
                spec["match_mode"] = match_mode
            if kind == "gpu_availability":
                min_visible = spec.get("min_visible")
                if not isinstance(min_visible, int) or min_visible <= 0:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{phase_id} entry {check_id} must define positive integer "
                        f"min_visible: {contract_path}"
                    )
            if kind == "scheduler_queue":
                max_running_jobs = spec.get("max_running_jobs")
                if not isinstance(max_running_jobs, int) or max_running_jobs <= 0:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} must define positive integer max_running_jobs: {contract_path}"
                    )
                max_queued_jobs = spec.get("max_queued_jobs")
                if max_queued_jobs is not None and (not isinstance(max_queued_jobs, int) or max_queued_jobs <= 0):
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{phase_id} entry {check_id} max_queued_jobs must be a positive integer when set: "
                        f"{contract_path}"
                    )
            specs.append(spec)
        resolved[phase_id] = tuple(specs)
    return resolved


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


def _validate_contract_path_ref(
    value: str,
    *,
    repo_root: Path,
    study_root: Path,
    label: str,
) -> None:
    resolve_path_ref(
        value,
        repo_root=repo_root,
        manifest_dir=study_root,
        default_base="manifest",
        label=label,
    )


def _discover_repo_root(study_root: Path) -> Path:
    for parent in (study_root, *study_root.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise ValueError(f"study record must live inside a dnadesign repository checkout: {study_root}")


__all__ = ["load_study_ops_contract"]
