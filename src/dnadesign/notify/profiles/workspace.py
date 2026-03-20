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

from dnadesign._contracts import (
    list_construct_workspaces_from_root,
    resolve_construct_workspace_config_path_from_root,
)

from ..errors import NotifyConfigError


@dataclass(frozen=True)
class ToolWorkspaceResolver:
    resolve_config: Callable[[str, Path, Path], Path]
    list_workspaces: Callable[[Path, Path], list[str]]


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


def _resolve_search_root(search_start: Path | None) -> Path:
    return (search_start or Path.cwd()).expanduser().resolve()


def _dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return tuple(deduped)


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
    search_root = _resolve_search_root(search_start)
    repo_root = _resolve_repo_root(search_root)
    names = resolver.list_workspaces(repo_root, search_root)
    return sorted(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))


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

    search_root = _resolve_search_root(search_start)
    repo_root = _resolve_repo_root(search_root)
    try:
        config_path = resolver.resolve_config(workspace_name, repo_root, search_root)
    except ValueError as exc:
        raise NotifyConfigError(str(exc)) from exc
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    config_resolved = config_path.expanduser().resolve()
    if config_resolved.exists() and config_resolved.is_file():
        return config_resolved

    available = list_tool_workspaces(tool=tool_name, search_start=search_root)
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


def _list_workspace_names_from_root(root: Path) -> list[str]:
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


def _resolve_config_from_workspace_root(workspace_name: str, repo_root: Path, relative_root: Path) -> Path:
    return _workspace_root(repo_root, relative_root) / workspace_name / "config.yaml"


def _list_workspace_names(repo_root: Path, relative_root: Path) -> list[str]:
    return _list_workspace_names_from_root(_workspace_root(repo_root, relative_root))


def _construct_workspace_roots(repo_root: Path, search_root: Path) -> tuple[Path, ...]:
    roots = [(repo_root / "src/dnadesign/construct/workspaces").resolve(), search_root]
    env_root = str(os.environ.get("CONSTRUCT_WORKSPACE_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root))
    return _dedupe_paths(roots)


def _resolve_construct_config_from_known_roots(workspace_name: str, repo_root: Path, search_root: Path) -> Path:
    missing_registry_error: str | None = None
    for root in _construct_workspace_roots(repo_root, search_root):
        try:
            return resolve_construct_workspace_config_path_from_root(
                workspaces_root=root,
                workspace_selector=workspace_name,
            )
        except ValueError as exc:
            message = str(exc)
            if message.startswith("construct workspace registry not found:"):
                if missing_registry_error is None:
                    missing_registry_error = message
                continue
            raise
    if missing_registry_error is not None:
        raise ValueError(missing_registry_error)
    return (search_root / workspace_name / "config.yaml").resolve()


def _list_construct_workspace_names(repo_root: Path, search_root: Path) -> list[str]:
    names: list[str] = []
    for root in _construct_workspace_roots(repo_root, search_root):
        names.extend(list_construct_workspaces_from_root(root))
    return sorted(dict.fromkeys(names))


def _infer_workspace_roots(repo_root: Path, search_root: Path) -> tuple[Path, ...]:
    roots = [(repo_root / "src/dnadesign/infer/workspaces").resolve(), (search_root / "workspaces").resolve()]
    env_root = str(os.environ.get("INFER_WORKSPACE_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root))
    return _dedupe_paths(roots)


def _resolve_infer_config_from_known_roots(workspace_name: str, repo_root: Path, search_root: Path) -> Path:
    roots = _infer_workspace_roots(repo_root, search_root)
    for root in roots:
        candidate = root / workspace_name / "config.yaml"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return (roots[0] / workspace_name / "config.yaml").resolve()


def _list_infer_workspace_names(repo_root: Path, search_root: Path) -> list[str]:
    names: list[str] = []
    for root in _infer_workspace_roots(repo_root, search_root):
        names.extend(_list_workspace_names_from_root(root))
    return sorted(dict.fromkeys(names))


register_tool_workspace_resolver(
    tool="construct",
    resolve_config=_resolve_construct_config_from_known_roots,
    list_workspaces=_list_construct_workspace_names,
)
register_tool_workspace_resolver(
    tool="densegen",
    resolve_config=lambda workspace_name, repo_root, search_root: _resolve_config_from_workspace_root(
        workspace_name, repo_root, Path("src/dnadesign/densegen/workspaces")
    ),
    list_workspaces=lambda repo_root, search_root: _list_workspace_names(
        repo_root,
        Path("src/dnadesign/densegen/workspaces"),
    ),
)
register_tool_workspace_resolver(
    tool="infer",
    resolve_config=_resolve_infer_config_from_known_roots,
    list_workspaces=_list_infer_workspace_names,
)
