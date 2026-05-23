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

from ..core.errors import NotifyConfigError


@dataclass(frozen=True)
class ToolWorkspaceResolver:
    resolve_config: Callable[[str, Path | None, Path], Path]
    list_workspaces: Callable[[Path | None, Path], list[str]]


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
    repo_root = _optional_repo_root(search_start)
    if repo_root is not None:
        return repo_root
    raise NotifyConfigError(
        "unable to determine repo root for --workspace mode; "
        "run from inside dnadesign repo, set DNADESIGN_REPO_ROOT, or pass --config explicitly"
    )


def _optional_repo_root(search_start: Path | None) -> Path | None:
    env_root = str(os.environ.get("DNADESIGN_REPO_ROOT") or "").strip()
    if env_root:
        repo_root = Path(env_root).expanduser().resolve()
        if not repo_root.exists() or not repo_root.is_dir():
            raise NotifyConfigError(f"DNADESIGN_REPO_ROOT is not a readable directory: {repo_root}")
        return repo_root
    start = (search_start or Path.cwd()).expanduser().resolve()
    return _repo_root_from(start)


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
    repo_root = _optional_repo_root(search_root)
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
    repo_root = _optional_repo_root(search_root)
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


def _construct_workspace_roots(repo_root: Path | None, search_root: Path) -> tuple[Path, ...]:
    roots = [search_root]
    if repo_root is not None:
        roots.append((repo_root / "src/dnadesign/construct/workspaces").resolve())
    env_root = str(os.environ.get("CONSTRUCT_WORKSPACE_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root))
    return _dedupe_paths(roots)


def _resolve_construct_config_from_known_roots(workspace_name: str, repo_root: Path | None, search_root: Path) -> Path:
    from dnadesign.construct import resolve_construct_workspace_config_path_from_root

    workspace_id, _, _ = workspace_name.partition(":")
    explicit_root = str(os.environ.get("CONSTRUCT_WORKSPACE_ROOT") or "").strip()
    if not explicit_root and search_root.name == workspace_id and (search_root / "construct.workspace.yaml").exists():
        return resolve_construct_workspace_config_path_from_root(
            workspaces_root=search_root.parent,
            workspace_selector=workspace_name,
        )
    roots = _construct_workspace_roots(repo_root, search_root)
    missing_registry_error: str | None = None
    for root in roots:
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
    if roots:
        return (roots[0] / workspace_id / "construct.workspace.yaml").resolve()
    if missing_registry_error is not None:
        raise ValueError(missing_registry_error)
    return (search_root / workspace_id / "construct.workspace.yaml").resolve()


def _list_construct_workspace_names(repo_root: Path | None, search_root: Path) -> list[str]:
    from dnadesign.construct import (
        list_construct_workspace_selectors,
        list_construct_workspace_selectors_from_root,
    )

    names: list[str] = []
    if (search_root / "construct.workspace.yaml").exists():
        names.extend(list_construct_workspace_selectors(search_root))
    for root in _construct_workspace_roots(repo_root, search_root):
        names.extend(list_construct_workspace_selectors_from_root(root))
    return sorted(dict.fromkeys(names))


def _densegen_workspace_roots(repo_root: Path | None, search_root: Path) -> tuple[Path, ...]:
    roots = [search_root]
    if repo_root is not None:
        roots.append((repo_root / "src/dnadesign/densegen/workspaces").resolve())
    env_root = str(os.environ.get("DENSEGEN_WORKSPACE_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root))
    return _dedupe_paths(roots)


def _resolve_densegen_config_from_known_roots(workspace_name: str, repo_root: Path | None, search_root: Path) -> Path:
    direct_config = search_root / "config.yaml"
    if search_root.name == workspace_name and direct_config.exists() and direct_config.is_file():
        return direct_config.resolve()
    roots = _densegen_workspace_roots(repo_root, search_root)
    for root in roots:
        candidate = root / workspace_name / "config.yaml"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return (roots[0] / workspace_name / "config.yaml").resolve()


def _list_densegen_workspace_names(repo_root: Path | None, search_root: Path) -> list[str]:
    names: list[str] = []
    direct_config = search_root / "config.yaml"
    if direct_config.exists() and direct_config.is_file():
        names.append(search_root.name)
    for root in _densegen_workspace_roots(repo_root, search_root):
        names.extend(_list_workspace_names_from_root(root))
    return sorted(dict.fromkeys(names))


def _infer_workspace_roots(repo_root: Path | None, search_root: Path) -> tuple[Path, ...]:
    search_root_resolved = search_root.resolve()
    if search_root_resolved.name == "workspaces":
        roots = [search_root_resolved]
    else:
        roots = [(search_root_resolved / "workspaces").resolve()]
    if repo_root is not None:
        roots.append((repo_root / "src/dnadesign/infer/workspaces").resolve())
    env_root = str(os.environ.get("INFER_WORKSPACE_ROOT") or "").strip()
    if env_root:
        roots.insert(0, Path(env_root))
    return _dedupe_paths(roots)


def _resolve_infer_config_from_known_roots(workspace_name: str, repo_root: Path | None, search_root: Path) -> Path:
    direct_config = search_root / "config.yaml"
    if search_root.name == workspace_name and direct_config.exists() and direct_config.is_file():
        return direct_config.resolve()
    roots = _infer_workspace_roots(repo_root, search_root)
    for root in roots:
        candidate = root / workspace_name / "config.yaml"
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return (roots[0] / workspace_name / "config.yaml").resolve()


def _list_infer_workspace_names(repo_root: Path | None, search_root: Path) -> list[str]:
    names: list[str] = []
    direct_config = search_root / "config.yaml"
    if direct_config.exists() and direct_config.is_file():
        names.append(search_root.name)
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
    resolve_config=_resolve_densegen_config_from_known_roots,
    list_workspaces=_list_densegen_workspace_names,
)
register_tool_workspace_resolver(
    tool="infer",
    resolve_config=_resolve_infer_config_from_known_roots,
    list_workspaces=_list_infer_workspace_names,
    aliases=("infer_evo2", "infer-evo2"),
)
