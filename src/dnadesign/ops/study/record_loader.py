"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/study/record_loader.py

Loads an OPS-facing study contract from an explicit workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import yaml

from dnadesign.ops.preflight import supported_preflight_check_kinds
from dnadesign.ops.status import resolve_path_ref

from .models import (
    STUDY_PREFLIGHT_SCOPES,
    STUDY_SUMMARY_SCOPES,
    StudyOpsContract,
    StudyPreflightContract,
)

_OPS_STUDY_TOP_LEVEL_KEYS = {
    "version",
    "study_id",
    "title",
    "parts",
    "ops_surfaces",
    "record_sources",
    "artifacts",
    "execution_surfaces",
    "snapshot",
    "preflight",
    "family",
}
_OPS_STUDY_PART_KEYS = {
    "artifacts",
    "execution_surfaces",
    "snapshot",
    "preflight",
}
_OPS_SURFACES_KEYS = {"status_kind", "preflight_kind"}
_LEGACY_PREFLIGHT_CHECK_KEYS = frozenset({"phase_id", "phase"})


def load_study_ops_contract(study_root: Path) -> StudyOpsContract:
    resolved_study_root = study_root.expanduser().resolve()
    repo_root = _discover_repo_root(resolved_study_root)
    contract_path = resolved_study_root / "operations" / "ops.study.yaml"
    if not contract_path.exists():
        raise ValueError(f"study record missing ops.study.yaml: {contract_path}")
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"ops.study.yaml must be a mapping: {contract_path}")
    payload = _expand_ops_study_parts(payload, contract_path=contract_path)
    _reject_unknown_mapping_keys(
        payload,
        allowed_keys=_OPS_STUDY_TOP_LEVEL_KEYS,
        label="ops.study.yaml",
        source=contract_path,
    )
    version = int(payload.get("version") or 0)
    if version != 2:
        raise ValueError(f"unsupported ops.study.yaml version {version}: {contract_path}")

    study_id = str(payload.get("study_id") or "").strip()
    title = _string_or_none(payload.get("title"))
    if not study_id:
        raise ValueError(f"ops.study.yaml must define study_id: {contract_path}")
    if "family" in payload:
        raise ValueError(
            "ops.study.yaml must not define legacy family; define explicit "
            f"ops_surfaces.status_kind and ops_surfaces.preflight_kind instead: {contract_path}"
        )
    ops_surfaces = payload.get("ops_surfaces")
    if ops_surfaces is None:
        ops_surfaces = {}
    if not isinstance(ops_surfaces, dict):
        raise ValueError(f"ops.study.yaml ops_surfaces must be a mapping: {contract_path}")
    _reject_unknown_mapping_keys(
        ops_surfaces,
        allowed_keys=_OPS_SURFACES_KEYS,
        label="ops.study.yaml ops_surfaces",
        source=contract_path,
    )
    status_kind = str(ops_surfaces.get("status_kind") or "").strip()
    preflight_kind = str(ops_surfaces.get("preflight_kind") or "").strip()
    if bool(status_kind) != bool(preflight_kind):
        raise ValueError(
            "ops.study.yaml ops_surfaces.status_kind and ops_surfaces.preflight_kind "
            f"must be declared together: {contract_path}"
        )
    resolved_status_kind = status_kind or None
    resolved_preflight_kind = preflight_kind or None

    record_sources = _validated_contract_refs_mapping(
        payload.get("record_sources"),
        repo_root=repo_root,
        study_root=resolved_study_root,
        contract_path=contract_path,
        label="ops.study.yaml record_sources",
    )

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
    checks_payload = preflight_payload.get("checks") or {}
    if checks_payload and not isinstance(checks_payload, dict):
        raise ValueError(f"ops.study.yaml preflight.checks must be a mapping: {contract_path}")
    scope_payloads = _validated_preflight_scopes(scopes_payload, contract_path=contract_path)
    forbidden = {
        "group_phase_bindings",
        "group_track_bindings",
        "next_scope",
    }.intersection(preflight_payload)
    if forbidden:
        raise ValueError(
            "ops.study.yaml preflight must use scopes.<scope>.include_groups, not "
            + ", ".join(sorted(forbidden))
            + f": {contract_path}"
        )
    declared_preflight_groups = _declared_check_groups(checks_payload)
    scope_groups = _validated_scope_groups(
        scope_payloads,
        declared_groups=declared_preflight_groups,
        contract_path=contract_path,
    )
    known_preflight_groups = set(declared_preflight_groups)
    unknown_scope_groups = sorted(
        group for groups in scope_groups.values() for group in groups if group not in known_preflight_groups
    )
    if unknown_scope_groups:
        raise ValueError(
            "ops.study.yaml preflight.scopes references unknown check_group(s) "
            + ", ".join(unknown_scope_groups)
            + f": {contract_path}"
        )
    check_specs = _validated_preflight_checks(
        checks_payload,
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
        status_kind=resolved_status_kind,
        preflight_kind=resolved_preflight_kind,
        title=title,
        snapshot_summary_scope=summary_scope,
        preflight=StudyPreflightContract(
            default_scope=default_scope,
            scope_groups=scope_groups,
            scope_payloads=scope_payloads,
            check_specs=check_specs,
        ),
        record_sources=record_sources,
        artifacts=artifacts,
        execution_surfaces=execution_surfaces,
        raw_payload=dict(payload),
    )


def _expand_ops_study_parts(payload: Mapping[str, object], *, contract_path: Path) -> dict[str, object]:
    parts_payload = payload.get("parts")
    if parts_payload is None:
        return dict(payload)
    if not isinstance(parts_payload, dict):
        raise ValueError(f"ops.study.yaml parts must be a mapping: {contract_path}")
    _reject_unknown_mapping_keys(
        parts_payload,
        allowed_keys=_OPS_STUDY_PART_KEYS,
        label="ops.study.yaml parts",
        source=contract_path,
    )
    expanded = {str(key): value for key, value in payload.items() if str(key) != "parts"}
    for raw_section, raw_ref in parts_payload.items():
        section = str(raw_section or "").strip()
        if not section:
            raise ValueError(f"ops.study.yaml parts keys must be non-empty: {contract_path}")
        if section in expanded:
            raise ValueError(f"ops.study.yaml parts.{section} duplicates an inline {section} section: {contract_path}")
        part_paths = _resolve_ops_study_part_paths(
            raw_ref,
            contract_path=contract_path,
            label=f"ops.study.yaml parts.{section}",
        )
        merged_mapping: dict[str, object] = {}
        for part_path in part_paths:
            part_payload = yaml.safe_load(part_path.read_text(encoding="utf-8")) or {}
            if not isinstance(part_payload, dict):
                raise ValueError(f"ops.study.yaml parts.{section} must load a mapping: {part_path}")
            _merge_ops_study_part_mapping(
                merged_mapping,
                part_payload,
                label=f"ops.study.yaml parts.{section}",
                part_path=part_path,
            )
        expanded[section] = merged_mapping
    return expanded


def _merge_ops_study_part_mapping(
    target: dict[str, object],
    source: Mapping[str, object],
    *,
    label: str,
    part_path: Path,
) -> None:
    for raw_key, value in source.items():
        key = str(raw_key or "").strip()
        if not key:
            raise ValueError(f"{label} loaded an empty key: {part_path}")
        if key not in target:
            target[key] = value
            continue
        existing = target[key]
        if isinstance(existing, dict) and isinstance(value, Mapping):
            _merge_ops_study_part_mapping(
                existing,
                value,
                label=f"{label}.{key}",
                part_path=part_path,
            )
            continue
        if isinstance(existing, list) and isinstance(value, list):
            existing.extend(value)
            continue
        raise ValueError(f"{label}.{key} is defined by multiple incompatible part files: {part_path}")


def _resolve_ops_study_part_paths(raw_value: object, *, contract_path: Path, label: str) -> tuple[Path, ...]:
    if isinstance(raw_value, list):
        if not raw_value:
            raise ValueError(f"{label} must list at least one path: {contract_path}")
        return tuple(
            _resolve_ops_study_part_path(item, contract_path=contract_path, label=f"{label}[{index}]")
            for index, item in enumerate(raw_value)
        )
    return (
        _resolve_ops_study_part_path(
            raw_value,
            contract_path=contract_path,
            label=label,
        ),
    )


def _resolve_ops_study_part_path(raw_value: object, *, contract_path: Path, label: str) -> Path:
    text = _string_or_none(raw_value)
    if text is None:
        raise ValueError(f"{label} must be a non-empty path: {contract_path}")
    if text.startswith(("repo:", "manifest:")):
        raise ValueError(f"{label} must be relative to the operations directory, not a path ref: {contract_path}")
    relative_path = Path(text)
    if relative_path.is_absolute():
        raise ValueError(f"{label} must be relative to the operations directory: {contract_path}")
    if ".." in relative_path.parts:
        raise ValueError(f"{label} must not escape the operations directory: {contract_path}")
    operations_dir = contract_path.parent.resolve()
    part_path = (operations_dir / relative_path).resolve()
    try:
        part_path.relative_to(operations_dir)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the operations directory: {contract_path}") from exc
    if part_path == contract_path.resolve():
        raise ValueError(f"{label} must not point back at ops.study.yaml: {contract_path}")
    if not part_path.exists():
        raise ValueError(f"{label} file does not exist: {part_path}")
    return part_path


def _string_or_none(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _reject_unknown_mapping_keys(
    payload: Mapping[str, object],
    *,
    allowed_keys: set[str],
    label: str,
    source: Path,
) -> None:
    unknown_keys = sorted(str(key) for key in payload if str(key) not in allowed_keys)
    if unknown_keys:
        raise ValueError(f"{label} contains unknown key(s) {', '.join(unknown_keys)}: {source}")


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


def _validated_scope_groups(
    scopes: Mapping[str, Mapping[str, object]],
    *,
    declared_groups: tuple[str, ...],
    contract_path: Path,
) -> dict[str, tuple[str, ...]]:
    required_scopes = STUDY_PREFLIGHT_SCOPES.difference(scopes)
    if required_scopes:
        raise ValueError(
            "ops.study.yaml preflight.scopes must define " + ", ".join(sorted(required_scopes)) + f": {contract_path}"
        )
    result: dict[str, tuple[str, ...]] = {}
    for scope, payload in scopes.items():
        unknown = set(payload).difference({"include_groups"})
        if unknown:
            raise ValueError(
                f"ops.study.yaml preflight.scopes.{scope} contains unknown key(s) "
                f"{', '.join(sorted(unknown))}: {contract_path}"
            )
        groups = _string_sequence(
            payload.get("include_groups") or [],
            label=f"ops.study.yaml preflight.scopes.{scope}.include_groups",
            source=contract_path,
        )
        if scope == "full":
            if groups != ("all",):
                raise ValueError(f"ops.study.yaml preflight.scopes.full.include_groups must be [all]: {contract_path}")
            result[scope] = declared_groups
            continue
        if "all" in groups:
            raise ValueError(
                f"ops.study.yaml preflight.scopes.{scope}.include_groups must list explicit groups: {contract_path}"
            )
        result[scope] = groups
    return result


def _declared_check_groups(checks_payload: Mapping[object, object]) -> tuple[str, ...]:
    groups: list[str] = []
    seen: set[str] = set()
    for raw_specs in checks_payload.values():
        if not isinstance(raw_specs, list):
            continue
        for raw_spec in raw_specs:
            if isinstance(raw_spec, dict):
                group = str(raw_spec.get("check_group") or "").strip()
                if group and group not in seen:
                    seen.add(group)
                    groups.append(group)
    return tuple(groups)


def _validated_preflight_checks(
    checks_payload: object,
    *,
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
    for raw_check_set_id, raw_specs in checks_payload.items():
        check_set_id = str(raw_check_set_id or "").strip()
        if not check_set_id:
            raise ValueError(f"ops.study.yaml preflight.checks keys must be non-empty: {contract_path}")
        if not isinstance(raw_specs, list):
            raise ValueError(f"ops.study.yaml preflight.checks.{check_set_id} must be a list: {contract_path}")
        specs: list[dict[str, object]] = []
        for index, raw_spec in enumerate(raw_specs, start=1):
            if not isinstance(raw_spec, dict):
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{check_set_id} entry {index} must be a mapping: {contract_path}"
                )
            spec = dict(raw_spec)
            legacy_keys = _LEGACY_PREFLIGHT_CHECK_KEYS.intersection(spec)
            if legacy_keys:
                raise ValueError(
                    f"ops.study.yaml preflight check {check_set_id}[{index}] contains legacy key(s) "
                    f"{', '.join(sorted(legacy_keys))}: {contract_path}"
                )
            kind = _string_or_none(spec.get("kind"))
            if kind is None:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{check_set_id} entry {index} must define kind: {contract_path}"
                )
            supported_kinds = supported_preflight_check_kinds()
            if kind not in supported_kinds:
                allowed_kinds = ", ".join(sorted(supported_kinds))
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{check_set_id} entry {index} has unsupported kind {kind!r}; "
                    f"expected one of: {allowed_kinds}: {contract_path}"
                )
            check_id = _string_or_none(spec.get("check_id"))
            if check_id is None:
                raise ValueError(
                    "ops.study.yaml preflight.checks."
                    f"{check_set_id} entry {index} must define check_id: {contract_path}"
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
                    f"{check_set_id} entry {check_id} must define check_group: {contract_path}"
                )
            if check_group not in known_groups:
                raise ValueError(
                    f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} references unknown check_group "
                    f"{check_group!r}: {contract_path}"
                )
            summary = _string_or_none(spec.get("summary"))
            if summary is None:
                raise ValueError(
                    "ops.study.yaml preflight.checks."
                    f"{check_set_id} entry {check_id} must define summary: {contract_path}"
                )
            required = spec.get("required")
            if required is not None and not isinstance(required, bool):
                raise ValueError(
                    "ops.study.yaml preflight.checks."
                    f"{check_set_id} entry {check_id} must use boolean required: {contract_path}"
                )

            if kind in {"path_exists", "dataset_snapshot", "sequence_view_contract"}:
                artifact = _string_or_none(spec.get("artifact"))
                if artifact is None:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} must define artifact: {contract_path}"
                    )
                if artifact not in artifact_ids:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} references unknown artifact "
                        f"{artifact!r}: {contract_path}"
                    )
            if kind == "sequence_view_contract":
                expected = spec.get("expected")
                if expected is not None and not isinstance(expected, dict):
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} expected must be a mapping when defined: {contract_path}"
                    )
            if kind == "dataset_snapshot":
                target_rows = spec.get("target_rows")
                if not isinstance(target_rows, int) or target_rows <= 0:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} must define positive integer "
                        f"target_rows: {contract_path}"
                    )
                row_count_mode = _string_or_none(spec.get("row_count_mode")) or "at_least"
                if row_count_mode not in {"at_least", "exact"}:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} has unsupported "
                        f"row_count_mode {row_count_mode!r}: {contract_path}"
                    )
            if kind in {
                "runbook_plan",
                "workspace_layout",
                "command",
                "scheduler_queue",
                "infer_sequence_view_completion",
            }:
                surface = _string_or_none(spec.get("surface"))
                if surface is None:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} must define surface: {contract_path}"
                    )
                if surface not in execution_surface_ids:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} references unknown surface "
                        f"{surface!r}: {contract_path}"
                    )
                surface_type = str((execution_surfaces.get(surface) or {}).get("surface_type") or "").strip()
                expected_surface_type = {
                    "runbook_plan": "runbook",
                    "workspace_layout": "workspace",
                    "command": "command",
                    "scheduler_queue": "scheduler",
                    "infer_sequence_view_completion": "command",
                }[kind]
                if surface_type != expected_surface_type:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} requires surface {surface!r} to use "
                        f"surface_type {expected_surface_type!r}: {contract_path}"
                    )
            if kind == "infer_sequence_view_completion":
                expected = spec.get("expected")
                if expected is not None and not isinstance(expected, dict):
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} expected must be a mapping when defined: {contract_path}"
                    )
            if kind == "environment":
                vars_payload = spec.get("vars")
                vars_list = _string_sequence(
                    vars_payload,
                    label=f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} vars",
                    source=contract_path,
                )
                spec["vars"] = list(vars_list)
                match_mode = _string_or_none(spec.get("match_mode")) or "all"
                if match_mode not in {"all", "any"}:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} match_mode must be one of: all, any: {contract_path}"
                    )
                spec["match_mode"] = match_mode
            if kind == "gpu_availability":
                min_visible = spec.get("min_visible")
                if not isinstance(min_visible, int) or min_visible <= 0:
                    raise ValueError(
                        f"ops.study.yaml preflight.checks.{check_set_id} entry {check_id} must define positive integer "
                        f"min_visible: {contract_path}"
                    )
            if kind == "scheduler_queue":
                max_running_jobs = spec.get("max_running_jobs")
                if not isinstance(max_running_jobs, int) or max_running_jobs <= 0:
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} must define positive integer "
                        f"max_running_jobs: {contract_path}"
                    )
                max_queued_jobs = spec.get("max_queued_jobs")
                if max_queued_jobs is not None and (not isinstance(max_queued_jobs, int) or max_queued_jobs <= 0):
                    raise ValueError(
                        "ops.study.yaml preflight.checks."
                        f"{check_set_id} entry {check_id} max_queued_jobs must be a positive integer when set: "
                        f"{contract_path}"
                    )
            specs.append(spec)
        resolved[check_set_id] = tuple(specs)
    return resolved


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
    raise ValueError(f"study record must live inside a repository with pyproject.toml: {study_root}")


__all__ = ["load_study_ops_contract"]
