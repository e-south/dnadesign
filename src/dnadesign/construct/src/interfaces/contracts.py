"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/interfaces/contracts.py

Public construct contracts for USR output resolution and workspace registry
lookup.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError as PydanticValidationError

from dnadesign.usr import resolve_usr_root_from_config

from ..contracts.errors import ConfigError, ConstructError
from ..workspaces.registry import load_workspace_registry, resolve_workspace_project_config_artifact_path

if TYPE_CHECKING:
    from .api import JobConfig

_WORKSPACES_ROOT = Path("src/dnadesign/construct/workspaces")
_WORKSPACE_REGISTRY_NAME = "construct.workspace.yaml"


@dataclass(frozen=True)
class ConstructUSROutputContract:
    config_path: Path
    usr_root: Path
    usr_dataset: str


def _required_mapping(raw: object, *, label: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    return raw


def _required_non_empty_string(raw: object, *, label: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError(f"{label} must be a non-empty string")
    return text


def _normalize_relative_dataset_path(dataset_value: object, *, label: str) -> str:
    dataset_raw = _required_non_empty_string(dataset_value, label=label)
    dataset_path = Path(dataset_raw.replace("\\", "/"))
    if dataset_path.is_absolute():
        raise ValueError(f"{label} must be a relative path")
    if any(part in {".", ".."} for part in dataset_path.parts):
        raise ValueError(f"{label} must not contain '.' or '..'")
    return Path(*dataset_path.parts).as_posix()


def _validate_construct_config_root(root: dict[str, object], *, config_path: Path) -> JobConfig:
    from .api import JobConfig

    try:
        return JobConfig.model_validate(root)
    except PydanticValidationError as exc:
        raise ValueError(f"Invalid config {config_path}: {exc}") from exc


def _load_construct_config(config_path: Path) -> tuple[Path, JobConfig]:
    from .api import load_job_config

    try:
        loaded, resolved_config_path = load_job_config(config_path)
    except (ConstructError, OSError) as exc:
        raise ValueError(str(exc)) from exc
    return resolved_config_path, loaded


def resolve_construct_usr_output_contract(
    config_path: Path,
    *,
    root: dict[str, object] | None = None,
) -> ConstructUSROutputContract:
    if root is None:
        resolved_config_path, loaded = _load_construct_config(config_path)
    else:
        resolved_config_path = config_path.expanduser().resolve()
        loaded = _validate_construct_config_root(
            _required_mapping(root, label="construct config"),
            config_path=resolved_config_path,
        )

    usr_dataset = _normalize_relative_dataset_path(loaded.job.output.target.dataset, label="job.output.target.dataset")
    root_value = loaded.job.output.target.root or loaded.job.input.source.root
    usr_root = resolve_usr_root_from_config(
        root_value,
        config_path=resolved_config_path,
        label="job.output.target.root or job.input.source.root",
    )
    if usr_root is None:
        raise ValueError("construct resolver requires job.input.source.root or job.output.target.root")

    return ConstructUSROutputContract(
        config_path=resolved_config_path,
        usr_root=usr_root,
        usr_dataset=usr_dataset,
    )


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


def _load_workspace_registry_contract(registry_path: Path):
    try:
        registry, _resolved_registry_path = load_workspace_registry(registry_path)
    except (ConfigError, OSError) as exc:
        raise ValueError(str(exc)) from exc
    return registry


def _workspace_projects(*, registry_path: Path, workspace_name: str):
    registry = _load_workspace_registry_contract(registry_path)
    projects = list(registry.workspace.projects)
    if not projects:
        raise ValueError(f"construct workspace '{workspace_name}' has no projects in {registry_path}")

    project_by_id: dict[str, object] = {}
    ordered_project_ids: list[str] = []
    for project in projects:
        resolved_project_id = _required_non_empty_string(project.id, label="workspace.projects[].id")
        if resolved_project_id in project_by_id:
            raise ValueError(f"construct workspace '{workspace_name}' has duplicate project id '{resolved_project_id}'")
        project_by_id[resolved_project_id] = project
        ordered_project_ids.append(resolved_project_id)
    return projects, ordered_project_ids, project_by_id


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


def _construct_workspace_selector_names(*, workspace_dir: Path) -> list[str]:
    registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
    try:
        registry = _load_workspace_registry_contract(registry_path)
    except ValueError:
        return []
    ordered_project_ids = [
        _required_non_empty_string(project.id, label="workspace.projects[].id")
        for project in registry.workspace.projects
    ]
    if not ordered_project_ids:
        return [workspace_dir.name]
    if len(ordered_project_ids) <= 1:
        return [workspace_dir.name]
    return [f"{workspace_dir.name}:{project_id}" for project_id in ordered_project_ids]


def list_construct_workspace_selectors(workspace: str | Path) -> list[str]:
    workspace_dir = Path(workspace).expanduser().resolve()
    if workspace_dir.name == _WORKSPACE_REGISTRY_NAME:
        workspace_dir = workspace_dir.parent
    registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
    if not registry_path.exists() or not registry_path.is_file():
        return []
    return _construct_workspace_selector_names(workspace_dir=workspace_dir)


def list_construct_workspace_selectors_from_root(workspaces_root: Path) -> list[str]:
    selectors: list[str] = []
    for workspace_name in list_construct_workspaces_from_root(workspaces_root):
        workspace_dir = Path(workspaces_root).resolve() / workspace_name
        selectors.extend(_construct_workspace_selector_names(workspace_dir=workspace_dir))
    return selectors


def resolve_construct_workspace_config_path_from_root(*, workspaces_root: Path, workspace_selector: str) -> Path:
    workspace_name, project_id = _parse_workspace_selector(workspace_selector)
    workspace_dir = (Path(workspaces_root).expanduser().resolve() / workspace_name).resolve()
    registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
    _projects, ordered_project_ids, project_by_id = _workspace_projects(
        registry_path=registry_path,
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

    try:
        config_path = resolve_workspace_project_config_artifact_path(
            workspace_dir=workspace_dir,
            config_value=project_cfg.artifacts.config.path,
        )
    except (ConfigError, OSError) as exc:
        raise ValueError(str(exc)) from exc
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


def _resolve_construct_workspace_match_from_config(config_path: Path) -> tuple[Path, str] | None:
    resolved_config_path = Path(config_path).expanduser().resolve()
    for workspace_dir in (resolved_config_path.parent, *resolved_config_path.parent.parents):
        registry_path = workspace_dir / _WORKSPACE_REGISTRY_NAME
        if not registry_path.exists():
            continue
        if not registry_path.is_file():
            raise ValueError(f"construct workspace registry is not a file: {registry_path}")

        projects, _ordered_project_ids, _project_by_id = _workspace_projects(
            registry_path=registry_path,
            workspace_name=workspace_dir.name,
        )

        matching_project_ids: list[str] = []
        for project_cfg in projects:
            project_id = _required_non_empty_string(project_cfg.id, label="workspace.projects[].id")
            try:
                candidate = resolve_workspace_project_config_artifact_path(
                    workspace_dir=workspace_dir,
                    config_value=project_cfg.artifacts.config.path,
                )
            except (ConfigError, OSError) as exc:
                raise ValueError(str(exc)) from exc
            if candidate == resolved_config_path:
                matching_project_ids.append(project_id)
        if not matching_project_ids:
            continue
        if len(matching_project_ids) > 1:
            joined = ", ".join(matching_project_ids)
            raise ValueError(
                "construct workspace "
                f"'{workspace_dir.name}' maps {resolved_config_path} "
                f"to multiple project ids: {joined}"
            )
        return workspace_dir, matching_project_ids[0]
    return None


def resolve_construct_workspace_root_from_config(config_path: Path) -> Path | None:
    match = _resolve_construct_workspace_match_from_config(config_path)
    if match is None:
        return None
    return match[0]


def resolve_construct_workspace_project_id_from_config(config_path: Path) -> str | None:
    match = _resolve_construct_workspace_match_from_config(config_path)
    if match is None:
        return None
    return match[1]


def resolve_construct_run_id_from_config(config_path: Path) -> str:
    _resolved_config_path, loaded = _load_construct_config(config_path)
    return f"construct-{loaded.job.id}"


__all__ = [
    "ConstructUSROutputContract",
    "list_construct_workspace_selectors",
    "list_construct_workspace_selectors_from_root",
    "list_construct_workspaces",
    "list_construct_workspaces_from_root",
    "resolve_construct_run_id_from_config",
    "resolve_construct_usr_output_contract",
    "resolve_construct_workspace_config_path_from_root",
    "resolve_construct_workspace_project_id_from_config",
    "resolve_construct_workspace_root_from_config",
]
