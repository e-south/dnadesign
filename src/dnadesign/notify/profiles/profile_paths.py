"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/profile_paths.py

Shared default profile path resolution for notify setup/watch resolver modes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from ..errors import NotifyConfigError
from .policy import default_profile_path_for_tool


def resolve_resolver_mode_profile_path(*, tool_name: str, config_path: Path) -> Path:
    resolved_config_path = config_path.expanduser().resolve()
    workspace_root = resolved_config_path.parent
    if tool_name == "infer":
        from dnadesign.infer.contracts import resolve_infer_notify_profile_path

        try:
            return resolve_infer_notify_profile_path(resolved_config_path)
        except ValueError as exc:
            raise NotifyConfigError(str(exc)) from exc
    if tool_name != "construct":
        return workspace_root / default_profile_path_for_tool(tool_name)
    from dnadesign.construct.contracts import (
        resolve_construct_workspace_project_id_from_config,
        resolve_construct_workspace_root_from_config,
    )

    try:
        resolved_workspace_root = resolve_construct_workspace_root_from_config(resolved_config_path)
        project_id = resolve_construct_workspace_project_id_from_config(resolved_config_path)
    except ValueError as exc:
        raise NotifyConfigError(str(exc)) from exc
    if resolved_workspace_root is not None:
        workspace_root = resolved_workspace_root
    default_path = workspace_root / default_profile_path_for_tool(tool_name)
    if project_id is None:
        return default_path
    return workspace_root / "outputs" / "notify" / "construct" / project_id / "profile.json"


__all__ = ["resolve_resolver_mode_profile_path"]
