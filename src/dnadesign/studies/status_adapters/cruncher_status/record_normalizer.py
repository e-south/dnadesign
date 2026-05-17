"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/status_adapters/cruncher_status/record_normalizer.py

Thin checked-in record normalization for Cruncher study status snapshots.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from dnadesign.ops.catalog import discover_repo_root
from dnadesign.ops.status import resolve_path_ref
from dnadesign.studies.core.models import StudyOpsContract
from dnadesign.studies.core.record_loader import load_study_ops_contract
from dnadesign.studies.core.record_locator import discover_active_study_selection

_REQUIRED_RECORD_FILES = (
    "campaign.yaml",
    "datasets.yaml",
    "status.md",
    "ops.study.yaml",
    "routes.md",
    "pipeline.yaml",
)
_BLOCKED_PHASE_STATUSES = frozenset({"blocked", "blocked_gpu"})
_RUNTIME_PHASE_STATUSES = frozenset({"ready", "planned", "in_progress"})


@dataclass(frozen=True)
class CruncherStudyCommandGroup:
    id: str
    purpose: str | None
    workspace_root: Path | None
    commands: tuple[str, ...]
    mutates_outputs: bool = False
    validation_role: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "purpose": self.purpose,
            "workspace_root": str(self.workspace_root) if self.workspace_root is not None else None,
            "commands": list(self.commands),
            "mutates_outputs": self.mutates_outputs,
            "validation_role": self.validation_role,
        }


@dataclass(frozen=True)
class CruncherStudyResolvedContext:
    requested_study_dir: Path | None
    resolved_study_dir: Path
    study_repo_root: Path
    study_id: str
    selection_source: str
    registry_path: Path | None
    active_study_id: str | None
    ops_contract: StudyOpsContract
    record_paths: dict[str, Path]
    missing_required_files: tuple[str, ...]
    status_excerpt: tuple[str, ...]
    pipeline_payload: dict[str, object]
    command_groups: tuple[CruncherStudyCommandGroup, ...]
    intent_payload: dict[str, object]
    native_agent_bootstrap: dict[str, object]
    phase_states: tuple[dict[str, object], ...]
    current_phase: str | None
    current_phase_is_known: bool
    next_ready_phase: dict[str, object] | None
    next_planned_phase: dict[str, object] | None
    blocked_phases: tuple[dict[str, object], ...]
    execution_surface_index: dict[str, Path]
    evidence: dict[str, object]


def resolve_cruncher_study_context(
    study_root: Path | str | None,
    *,
    repo_root: Path | None,
    status_kind: str,
) -> CruncherStudyResolvedContext:
    requested_study_dir = Path(study_root).expanduser().resolve() if study_root is not None else None
    selection_source = "explicit"
    registry_path: Path | None = None
    active_study_id: str | None = None

    if requested_study_dir is None:
        selection = discover_active_study_selection(repo_root=repo_root, status_kind=status_kind)
        resolved_study_dir = selection.study_root
        selection_source = "active_registry"
        registry_path = selection.index_path
        active_study_id = selection.active_study_id
        resolved_repo_root = selection.repo_root
    else:
        resolved_study_dir = requested_study_dir
        resolved_repo_root = (
            repo_root.expanduser().resolve() if repo_root is not None else discover_repo_root(resolved_study_dir)
        )
        if resolved_repo_root is None:
            raise ValueError(f"status kind '{status_kind}' requires --study-dir inside a dnadesign repository checkout")
        candidate_registry_path = resolved_repo_root / "docs" / "studies" / "index.yaml"
        if candidate_registry_path.exists():
            registry_path = candidate_registry_path
            payload = yaml.safe_load(candidate_registry_path.read_text(encoding="utf-8")) or {}
            if isinstance(payload, dict):
                active_study_id = _string_or_none(payload.get("active_study_id"))

    ops_contract = load_study_ops_contract(resolved_study_dir)
    record_paths = {name: resolved_study_dir / name for name in _REQUIRED_RECORD_FILES}
    missing_required_files = tuple(name for name, path in record_paths.items() if not path.exists())
    status_excerpt = _load_status_excerpt(record_paths["status.md"])
    pipeline_payload = _load_yaml_mapping(record_paths["pipeline.yaml"], label="study pipeline")
    command_groups = _load_command_groups(
        pipeline_payload.get("command_groups"),
        repo_root=resolved_repo_root,
        study_root=resolved_study_dir,
        label="pipeline.yaml command_groups",
    )
    intent_payload = _normalized_intent_payload(
        pipeline_payload.get("intent"),
        repo_root=resolved_repo_root,
        study_root=resolved_study_dir,
    )
    native_agent_bootstrap = _normalized_native_agent_bootstrap(
        pipeline_payload.get("native_agent_bootstrap"),
        repo_root=resolved_repo_root,
        study_root=resolved_study_dir,
    )
    phase_states = ops_contract.phase_states
    current_phase = ops_contract.current_phase_id
    current_phase_is_known = current_phase in {str(phase.get("id") or "").strip() for phase in phase_states}
    next_ready_phase = _first_phase_with_status(phase_states, statuses={"ready"})
    next_planned_phase = _next_runtime_phase_after_current(
        phase_states,
        current_phase=current_phase,
        statuses=_RUNTIME_PHASE_STATUSES,
    )
    blocked_phases = tuple(
        phase for phase in phase_states if str(phase.get("status") or "").strip() in _BLOCKED_PHASE_STATUSES
    )
    execution_surface_index = _resolve_execution_surface_index(
        ops_contract.execution_surfaces,
        repo_root=resolved_repo_root,
        study_root=resolved_study_dir,
    )
    evidence = {
        "study_id": resolved_study_dir.name,
        "selection_source": selection_source,
        "registry_path": str(registry_path) if registry_path is not None else None,
        "active_study_id": active_study_id,
        "record_root": str(resolved_study_dir),
    }
    return CruncherStudyResolvedContext(
        requested_study_dir=requested_study_dir,
        resolved_study_dir=resolved_study_dir,
        study_repo_root=resolved_repo_root,
        study_id=resolved_study_dir.name,
        selection_source=selection_source,
        registry_path=registry_path,
        active_study_id=active_study_id,
        ops_contract=ops_contract,
        record_paths=record_paths,
        missing_required_files=missing_required_files,
        status_excerpt=status_excerpt,
        pipeline_payload=pipeline_payload,
        command_groups=command_groups,
        intent_payload=intent_payload,
        native_agent_bootstrap=native_agent_bootstrap,
        phase_states=phase_states,
        current_phase=current_phase,
        current_phase_is_known=current_phase_is_known,
        next_ready_phase=next_ready_phase,
        next_planned_phase=next_planned_phase,
        blocked_phases=blocked_phases,
        execution_surface_index=execution_surface_index,
        evidence=evidence,
    )


def _load_yaml_mapping(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping: {path}")
    return dict(payload)


def _load_status_excerpt(path: Path, *, max_lines: int = 8) -> tuple[str, ...]:
    if not path.exists():
        return ()
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    preferred = _extract_status_section_excerpt(
        raw_lines,
        section_titles={
            "current phase and surfaces",
            "current phase",
            "current route and surfaces",
            "current route",
            "current track and surfaces",
            "current track",
        },
        max_lines=max_lines,
    )
    if preferred:
        return preferred
    return _fallback_status_excerpt(raw_lines, max_lines=max_lines)


def _extract_status_section_excerpt(
    raw_lines: Sequence[str],
    *,
    section_titles: set[str],
    max_lines: int,
) -> tuple[str, ...]:
    lines: list[str] = []
    in_section = False
    for raw_line in raw_lines:
        line = raw_line.strip()
        heading = _markdown_heading_title(line)
        if heading is not None:
            if in_section:
                break
            if heading in section_titles:
                in_section = True
            continue
        if not in_section:
            continue
        if not _should_include_status_excerpt_line(line):
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return tuple(lines)


def _fallback_status_excerpt(raw_lines: Sequence[str], *, max_lines: int) -> tuple[str, ...]:
    lines: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not _should_include_status_excerpt_line(line):
            continue
        lines.append(line)
        if len(lines) >= max_lines:
            break
    return tuple(lines)


def _markdown_heading_title(line: str) -> str | None:
    if not line.startswith("#"):
        return None
    return line.lstrip("#").strip().lower() or None


def _should_include_status_excerpt_line(line: str) -> bool:
    metadata_prefixes = ("**Owner:**", "**Last verified:**")
    if not line:
        return False
    if line.startswith("#"):
        return False
    if line.startswith(metadata_prefixes):
        return False
    return True


def _load_command_groups(
    payload: object,
    *,
    repo_root: Path,
    study_root: Path,
    label: str,
) -> tuple[CruncherStudyCommandGroup, ...]:
    if payload is None:
        return ()
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a list: {study_root / 'pipeline.yaml'}")
    groups: list[CruncherStudyCommandGroup] = []
    seen_group_ids: set[str] = set()
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label} entry {index} must be a mapping: {study_root / 'pipeline.yaml'}")
        group_id = _required_text(item.get("id"), label=f"{label}[{index}].id", source=study_root / "pipeline.yaml")
        if group_id in seen_group_ids:
            raise ValueError(f"{label} must not duplicate id {group_id!r}: {study_root / 'pipeline.yaml'}")
        seen_group_ids.add(group_id)
        commands = _string_sequence(
            item.get("commands"),
            label=f"{label}[{group_id}].commands",
            source=study_root / "pipeline.yaml",
        )
        raw_workspace_ref = _string_or_none(item.get("workspace_ref"))
        workspace_root = (
            resolve_path_ref(
                raw_workspace_ref,
                repo_root=repo_root,
                manifest_dir=study_root,
                default_base="manifest",
                label=f"{label}[{group_id}].workspace_ref",
            )
            if raw_workspace_ref is not None
            else None
        )
        groups.append(
            CruncherStudyCommandGroup(
                id=group_id,
                purpose=_string_or_none(item.get("purpose")),
                workspace_root=workspace_root,
                commands=commands,
                mutates_outputs=bool(item.get("mutates_outputs", False)),
                validation_role=_string_or_none(item.get("validation_role")),
            )
        )
    return tuple(groups)


def _normalized_intent_payload(
    payload: object,
    *,
    repo_root: Path,
    study_root: Path,
) -> dict[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline.yaml intent must be a mapping: {study_root / 'pipeline.yaml'}")
    normalized = dict(payload)
    normalized["context_refs"] = _resolve_path_ref_sequence(
        payload.get("context_refs"),
        repo_root=repo_root,
        study_root=study_root,
        label="pipeline.yaml intent.context_refs",
    )
    normalized["decision_refs"] = _resolve_path_ref_sequence(
        payload.get("decision_refs"),
        repo_root=repo_root,
        study_root=study_root,
        label="pipeline.yaml intent.decision_refs",
    )
    return normalized


def _normalized_native_agent_bootstrap(
    payload: object,
    *,
    repo_root: Path,
    study_root: Path,
) -> dict[str, object]:
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"pipeline.yaml native_agent_bootstrap must be a mapping: {study_root / 'pipeline.yaml'}")
    normalized = dict(payload)
    normalized["open_first"] = _resolve_path_ref_sequence(
        payload.get("open_first"),
        repo_root=repo_root,
        study_root=study_root,
        label="pipeline.yaml native_agent_bootstrap.open_first",
    )
    normalized["must_preserve"] = list(
        _string_sequence(
            payload.get("must_preserve") or [],
            label="pipeline.yaml native_agent_bootstrap.must_preserve",
            source=study_root / "pipeline.yaml",
            allow_empty=True,
        )
    )
    return normalized


def _resolve_path_ref_sequence(
    payload: object,
    *,
    repo_root: Path,
    study_root: Path,
    label: str,
) -> list[str]:
    if payload is None:
        return []
    values = _string_sequence(payload, label=label, source=study_root / "pipeline.yaml", allow_empty=True)
    return [
        str(
            resolve_path_ref(
                value,
                repo_root=repo_root,
                manifest_dir=study_root,
                default_base="manifest",
                label=label,
            )
        )
        for value in values
    ]


def _resolve_execution_surface_index(
    execution_surfaces: Mapping[str, Mapping[str, object]],
    *,
    repo_root: Path,
    study_root: Path,
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for surface_id, payload in execution_surfaces.items():
        surface_type = _string_or_none(payload.get("surface_type"))
        if surface_type == "workspace":
            ref = _required_text(payload.get("workspace_ref"), label=f"{surface_id}.workspace_ref", source=study_root)
            resolved[surface_id] = resolve_path_ref(
                ref,
                repo_root=repo_root,
                manifest_dir=study_root,
                default_base="repo",
                label=f"{surface_id}.workspace_ref",
            )
        elif surface_type == "runbook":
            ref = _required_text(payload.get("runbook_ref"), label=f"{surface_id}.runbook_ref", source=study_root)
            resolved[surface_id] = resolve_path_ref(
                ref,
                repo_root=repo_root,
                manifest_dir=study_root,
                default_base="repo",
                label=f"{surface_id}.runbook_ref",
            )
    return resolved


def _first_phase_with_status(
    phase_states: Sequence[Mapping[str, object]],
    *,
    statuses: set[str],
) -> dict[str, object] | None:
    for phase in phase_states:
        if str(phase.get("status") or "").strip() in statuses:
            return dict(phase)
    return None


def _next_runtime_phase_after_current(
    phase_states: Sequence[Mapping[str, object]],
    *,
    current_phase: str | None,
    statuses: set[str],
) -> dict[str, object] | None:
    current_phase_id = _string_or_none(current_phase)
    current_index = None
    if current_phase_id is not None:
        for index, phase in enumerate(phase_states):
            if str(phase.get("id") or "").strip() == current_phase_id:
                current_index = index
                break
    start_index = (current_index + 1) if current_index is not None else 0
    for phase in phase_states[start_index:]:
        if str(phase.get("status") or "").strip() in statuses:
            return dict(phase)
    return None


def _required_text(value: object, *, label: str, source: Path) -> str:
    text = _string_or_none(value)
    if text is None:
        raise ValueError(f"{label} is required in {source}")
    return text


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
    for index, raw_value in enumerate(values, start=1):
        text = _string_or_none(raw_value)
        if text is None:
            raise ValueError(f"{label} entry {index} must be non-empty: {source}")
        items.append(text)
    if not items and not allow_empty:
        raise ValueError(f"{label} must not be empty: {source}")
    return tuple(items)


__all__ = [
    "CruncherStudyCommandGroup",
    "CruncherStudyResolvedContext",
    "resolve_cruncher_study_context",
]
