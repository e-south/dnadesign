"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/_contracts/construct_workspace.py

Shared construct workspace registry parsing for external tool integrations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

_WORKSPACES_ROOT = Path("src/dnadesign/construct/workspaces")
_WORKSPACE_REGISTRY_NAME = "construct.workspace.yaml"


def _required_mapping(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _required_non_empty_string(raw: object, *, label: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _construct_workspaces_root(repo_root: Path) -> Path:
    env_root = str(os.environ.get("CONSTRUCT_WORKSPACE_ROOT") or "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (repo_root / _WORKSPACES_ROOT).resolve()


def _parse_workspace_selector(workspace_selector: str) -> tuple[str, str | None]:
    raw = str(workspace_selector or "").strip()
    if not raw:
        raise ValueError("workspace selector must be a non-empty string")
    workspace_name, separator, project_id = raw.partition(":")
    workspace_name = workspace_name.strip()
    if not workspace_name:
        raise ValueError("workspace selector must start with a workspace name")
    if not separator:
        return workspace_name, None
    project_name = project_id.strip()
    if not project_name:
        raise ValueError("workspace selector project id cannot be empty after ':'")
    return workspace_name, project_name


def _resolve_workspace_relative_path(*, workspace_dir: Path, value: object, label: str) -> Path:
    text = _required_non_empty_string(value, label=label)
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        raise ValueError(f"{label} must be workspace-relative, not absolute")
    resolved = (workspace_dir / candidate).resolve()
    try:
        resolved.relative_to(workspace_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside the workspace root: {text}") from exc
    return resolved


def _workspace_registry_payload(registry_path: Path) -> dict[str, object]:
    if not registry_path.exists():
        raise ValueError(f"construct workspace registry not found: {registry_path}")
    if not registry_path.is_file():
        raise ValueError(f"construct workspace registry is not a file: {registry_path}")
    try:
        raw = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise ValueError(f"failed to parse construct workspace registry '{registry_path}': {exc}") from exc
    return _required_mapping(raw, label="construct workspace registry")


def _workspace_projects(
    *, registry_path: Path, payload: dict[str, object], workspace_name: str
) -> tuple[list[dict[str, object]], list[str], dict[str, dict[str, object]]]:
    workspace_cfg = _required_mapping(payload.get("workspace"), label="workspace")
    projects = workspace_cfg.get("projects")
    if not isinstance(projects, list):
        raise ValueError(f"construct workspace registry {registry_path} must define workspace.projects as a list")
    if not projects:
        raise ValueError(f"construct workspace '{workspace_name}' has no projects in {registry_path}")

    project_by_id: dict[str, dict[str, object]] = {}
    ordered_project_ids: list[str] = []
    normalized_projects: list[dict[str, object]] = []
    for idx, project in enumerate(projects):
        project_cfg = _required_mapping(project, label=f"workspace.projects[{idx}]")
        resolved_project_id = _required_non_empty_string(project_cfg.get("id"), label=f"workspace.projects[{idx}].id")
        if resolved_project_id in project_by_id:
            raise ValueError(f"construct workspace '{workspace_name}' has duplicate project id '{resolved_project_id}'")
        project_by_id[resolved_project_id] = project_cfg
        ordered_project_ids.append(resolved_project_id)
        normalized_projects.append(project_cfg)
    return normalized_projects, ordered_project_ids, project_by_id


def list_construct_workspaces_from_root(workspaces_root: Path) -> list[str]:
    workspaces_root = Path(workspaces_root).expanduser().resolve()
    if not workspaces_root.exists() or not workspaces_root.is_dir():
        return []
    names: list[str] = []
    for candidate in sorted(workspaces_root.iterdir()):
        if not candidate.is_dir():
            continue
        registry_path = candidate / _WORKSPACE_REGISTRY_NAME
        if registry_path.exists() and registry_path.is_file():
            names.append(candidate.name)
    return names


def list_construct_workspaces(repo_root: Path) -> list[str]:
    return list_construct_workspaces_from_root(_construct_workspaces_root(repo_root))


def resolve_construct_workspace_config_path_from_root(*, workspaces_root: Path, workspace_selector: str) -> Path:
    workspace_name, project_id = _parse_workspace_selector(workspace_selector)
    workspace_dir = (Path(workspaces_root).expanduser().resolve() / workspace_name).resolve()
    registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
    payload = _workspace_registry_payload(registry_path)
    _projects, ordered_project_ids, project_by_id = _workspace_projects(
        registry_path=registry_path,
        payload=payload,
        workspace_name=workspace_name,
    )

    resolved_project_id = project_id
    if resolved_project_id is None:
        if len(ordered_project_ids) == 1:
            resolved_project_id = ordered_project_ids[0]
        else:
            available = ", ".join(ordered_project_ids)
            raise ValueError(
                f"construct workspace '{workspace_name}' has multiple projects. "
                f"Pass --workspace {workspace_name}:<project-id> or --config explicitly. "
                f"Available project ids: {available}"
            )

    project_cfg = project_by_id.get(resolved_project_id)
    if project_cfg is None:
        available = ", ".join(ordered_project_ids)
        raise ValueError(
            f"construct workspace '{workspace_name}' does not define project '{resolved_project_id}'. "
            f"Available project ids: {available}"
        )

    config_path = _resolve_workspace_relative_path(
        workspace_dir=workspace_dir,
        value=project_cfg.get("config"),
        label=f"workspace.projects[{resolved_project_id}].config",
    )
    if not config_path.exists():
        raise ValueError(
            f"construct workspace '{workspace_name}' project '{resolved_project_id}' config not found: {config_path}"
        )
    if not config_path.is_file():
        raise ValueError(
            "construct workspace "
            f"'{workspace_name}' project '{resolved_project_id}' config is not a file: {config_path}"
        )
    return config_path


def resolve_construct_workspace_config_path(*, repo_root: Path, workspace_selector: str) -> Path:
    return resolve_construct_workspace_config_path_from_root(
        workspaces_root=_construct_workspaces_root(repo_root),
        workspace_selector=workspace_selector,
    )


def resolve_construct_workspace_project_id_from_config(config_path: Path) -> str | None:
    resolved_config_path = Path(config_path).expanduser().resolve()
    workspace_dir = resolved_config_path.parent
    registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
    if not registry_path.exists() or not registry_path.is_file():
        return None

    payload = _workspace_registry_payload(registry_path)
    projects, _ordered_project_ids, _project_by_id = _workspace_projects(
        registry_path=registry_path,
        payload=payload,
        workspace_name=workspace_dir.name,
    )

    matching_project_ids: list[str] = []
    for idx, project_cfg in enumerate(projects):
        project_id = _required_non_empty_string(project_cfg.get("id"), label=f"workspace.projects[{idx}].id")
        candidate = _resolve_workspace_relative_path(
            workspace_dir=workspace_dir,
            value=project_cfg.get("config"),
            label=f"workspace.projects[{project_id}].config",
        )
        if candidate == resolved_config_path:
            matching_project_ids.append(project_id)
    if not matching_project_ids:
        return None
    if len(matching_project_ids) > 1:
        joined = ", ".join(matching_project_ids)
        raise ValueError(
            f"construct workspace '{workspace_dir.name}' maps {resolved_config_path} to multiple project ids: {joined}"
        )
    return matching_project_ids[0]
