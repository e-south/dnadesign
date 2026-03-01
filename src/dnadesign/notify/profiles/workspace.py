"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/workspace.py

Workspace-to-config resolvers for notify setup/watch shorthand flows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..errors import NotifyConfigError


@dataclass(frozen=True)
class ToolWorkspaceResolver:
    resolve_config: Callable[[str, Path], Path]
    list_workspaces: Callable[[Path], list[str]]


_TOOL_WORKSPACE_RESOLVERS: dict[str, ToolWorkspaceResolver] = {}
_TOOL_WORKSPACE_ALIASES: dict[str, str] = {}


def _normalize_name(value: str | None, *, field: str) -> str:
    if value is None:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    name = str(value).strip().lower()
    if not name:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return name


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _resolve_repo_root(search_start: Path | None) -> Path:
    env_root = str(os.environ.get("DNADESIGN_REPO_ROOT") or "").strip()
    if env_root:
        repo_root = Path(env_root).expanduser().resolve()
        if not repo_root.exists() or not repo_root.is_dir():
            raise NotifyConfigError(f"DNADESIGN_REPO_ROOT is not a readable directory: {repo_root}")
        return repo_root
    start = (search_start or Path.cwd()).expanduser().resolve()
    repo_root = _repo_root_from(start)
    if repo_root is None:
        raise NotifyConfigError(
            "unable to determine repo root for --workspace mode; "
            "run from inside dnadesign repo, set DNADESIGN_REPO_ROOT, or pass --config explicitly"
        )
    return repo_root


def register_tool_workspace_resolver(
    *,
    tool: str,
    resolve_config: Callable[[str, Path], Path],
    list_workspaces: Callable[[Path], list[str]],
    aliases: tuple[str, ...] = (),
) -> None:
    tool_name = _normalize_name(tool, field="tool")
    if tool_name in _TOOL_WORKSPACE_RESOLVERS:
        raise NotifyConfigError(f"tool '{tool_name}' is already registered")
    if not callable(resolve_config):
        raise NotifyConfigError("resolve_config must be callable")
    if not callable(list_workspaces):
        raise NotifyConfigError("list_workspaces must be callable")

    alias_names: list[str] = []
    for alias in aliases:
        alias_name = _normalize_name(alias, field="alias")
        if alias_name == tool_name:
            raise NotifyConfigError(f"alias '{alias_name}' cannot equal tool name '{tool_name}'")
        if alias_name in _TOOL_WORKSPACE_ALIASES or alias_name in _TOOL_WORKSPACE_RESOLVERS:
            raise NotifyConfigError(f"alias '{alias_name}' is already registered")
        alias_names.append(alias_name)

    _TOOL_WORKSPACE_RESOLVERS[tool_name] = ToolWorkspaceResolver(
        resolve_config=resolve_config,
        list_workspaces=list_workspaces,
    )
    for alias_name in alias_names:
        _TOOL_WORKSPACE_ALIASES[alias_name] = tool_name


def normalize_tool_name(tool: str | None) -> str | None:
    if tool is None:
        return None
    value = _normalize_name(tool, field="tool")
    return _TOOL_WORKSPACE_ALIASES.get(value, value)


def list_tool_workspaces(*, tool: str, search_start: Path | None = None) -> list[str]:
    tool_name = normalize_tool_name(tool)
    if tool_name is None:
        raise NotifyConfigError("tool must be a non-empty string when provided")
    resolver = _TOOL_WORKSPACE_RESOLVERS.get(tool_name)
    if resolver is None:
        allowed = ", ".join(sorted(_TOOL_WORKSPACE_RESOLVERS))
        raise NotifyConfigError(f"unsupported tool '{tool}'. Supported values: {allowed}")
    repo_root = _resolve_repo_root(search_start)
    return sorted(dict.fromkeys(str(name).strip() for name in resolver.list_workspaces(repo_root) if str(name).strip()))


def resolve_tool_workspace_config_path(*, tool: str, workspace: str, search_start: Path | None = None) -> Path:
    tool_name = normalize_tool_name(tool)
    if tool_name is None:
        raise NotifyConfigError("tool must be a non-empty string when provided")
    resolver = _TOOL_WORKSPACE_RESOLVERS.get(tool_name)
    if resolver is None:
        allowed = ", ".join(sorted(_TOOL_WORKSPACE_RESOLVERS))
        raise NotifyConfigError(f"unsupported tool '{tool}'. Supported values: {allowed}")

    workspace_name = str(workspace or "").strip()
    if not workspace_name:
        raise NotifyConfigError("workspace must be a non-empty string when provided")
    if any(ch in workspace_name for ch in ("/", "\\")):
        raise NotifyConfigError("workspace must be a workspace name (not a path); pass --config for explicit paths")

    repo_root = _resolve_repo_root(search_start)
    config_path = resolver.resolve_config(workspace_name, repo_root)
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    config_resolved = config_path.expanduser().resolve()
    if config_resolved.exists() and config_resolved.is_file():
        return config_resolved

    available = list_tool_workspaces(tool=tool_name, search_start=repo_root)
    if available:
        available_text = ", ".join(available[:12])
        if len(available) > 12:
            available_text += ", ..."
        raise NotifyConfigError(
            f"workspace '{workspace_name}' not found for tool '{tool_name}' at {config_resolved}. "
            f"Available workspaces: {available_text}"
        )
    raise NotifyConfigError(f"workspace '{workspace_name}' not found for tool '{tool_name}' at {config_resolved}")


def _workspace_root(repo_root: Path, relative_root: Path) -> Path:
    return (repo_root / relative_root).resolve()


def _resolve_config_from_workspace_root(workspace_name: str, repo_root: Path, relative_root: Path) -> Path:
    return _workspace_root(repo_root, relative_root) / workspace_name / "config.yaml"


def _list_workspace_names(repo_root: Path, relative_root: Path) -> list[str]:
    root = _workspace_root(repo_root, relative_root)
    if not root.exists() or not root.is_dir():
        return []
    names: list[str] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        config = candidate / "config.yaml"
        if config.exists() and config.is_file():
            names.append(candidate.name)
    return sorted(names)


register_tool_workspace_resolver(
    tool="densegen",
    resolve_config=lambda workspace_name, repo_root: _resolve_config_from_workspace_root(
        workspace_name, repo_root, Path("src/dnadesign/densegen/workspaces")
    ),
    list_workspaces=lambda repo_root: _list_workspace_names(repo_root, Path("src/dnadesign/densegen/workspaces")),
)
register_tool_workspace_resolver(
    tool="infer_evo2",
    resolve_config=lambda workspace_name, repo_root: _resolve_config_from_workspace_root(
        workspace_name, repo_root, Path("src/dnadesign/infer/workspaces")
    ),
    list_workspaces=lambda repo_root: _list_workspace_names(repo_root, Path("src/dnadesign/infer/workspaces")),
    aliases=("infer-evo2",),
)
