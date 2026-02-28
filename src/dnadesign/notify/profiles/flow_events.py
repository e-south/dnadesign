"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/profiles/flow_events.py

Setup event-source resolution helpers for notify profile setup workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..errors import NotifyConfigError
from ..events.source import normalize_tool_name, resolve_tool_events_path
from .flow_types import SetupEventsResolution
from .workspace import resolve_tool_workspace_config_path


def resolve_setup_events(
    *,
    events: Path | None,
    tool: str | None,
    config: Path | None,
    workspace: str | None,
    policy: str | None,
    search_start: Path | None = None,
    resolve_tool_events_path_fn: Callable[..., tuple[Path, str | None]] = resolve_tool_events_path,
    resolve_tool_workspace_config_fn: Callable[..., Path] = resolve_tool_workspace_config_path,
    normalize_tool_name_fn: Callable[[str | None], str | None] = normalize_tool_name,
) -> SetupEventsResolution:
    has_events = events is not None
    has_tool = tool is not None or config is not None or workspace is not None
    if has_events and has_tool:
        raise NotifyConfigError("pass either --events or --tool with --config/--workspace, not both")
    if not has_events and not has_tool:
        raise NotifyConfigError("pass either --events or --tool with --config/--workspace")

    if has_events:
        events_path = events if events is not None else Path("")
        return SetupEventsResolution(
            events_path=events_path,
            events_source=None,
            policy=policy,
            tool_name=None,
            events_require_exists=True,
        )

    has_config = config is not None
    has_workspace = workspace is not None
    if tool is None or has_config == has_workspace:
        raise NotifyConfigError("resolver mode requires --tool with exactly one of --config or --workspace")
    if has_workspace:
        config_path = resolve_tool_workspace_config_fn(
            tool=tool,
            workspace=str(workspace),
            search_start=search_start,
        )
        if not isinstance(config_path, Path):
            config_path = Path(config_path)
        config_path = config_path.expanduser().resolve()
    else:
        config_path = config.expanduser().resolve() if config is not None else Path("")
    events_path, default_policy = resolve_tool_events_path_fn(tool=tool, config=config_path)
    tool_name = normalize_tool_name_fn(tool)
    events_source = {
        "tool": str(tool_name),
        "config": str(config_path),
    }
    policy_value = policy if policy is not None else default_policy
    return SetupEventsResolution(
        events_path=events_path,
        events_source=events_source,
        policy=policy_value,
        tool_name=tool_name,
        events_require_exists=False,
    )
