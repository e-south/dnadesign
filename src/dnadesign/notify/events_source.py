"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/events_source.py

Tool config resolvers for expected USR events log destinations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .errors import NotifyConfigError
from .events_source_builtin import register_builtin_tool_events_sources


@dataclass(frozen=True)
class ToolEventsSourceResolver:
    resolver: Callable[[Path], Path]
    default_policy: str | None


_TOOL_RESOLVERS: dict[str, ToolEventsSourceResolver] = {}
_TOOL_ALIASES: dict[str, str] = {}


def _normalize_name(value: str | None, *, field: str) -> str:
    if value is None:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    name = str(value).strip().lower()
    if not name:
        raise NotifyConfigError(f"{field} must be a non-empty string when provided")
    return name


def register_tool_events_source(
    *,
    tool: str,
    resolver: Callable[[Path], Path],
    default_policy: str | None = None,
    aliases: tuple[str, ...] = (),
) -> None:
    tool_name = _normalize_name(tool, field="tool")
    if tool_name in _TOOL_RESOLVERS:
        raise NotifyConfigError(f"tool '{tool_name}' is already registered")
    if not callable(resolver):
        raise NotifyConfigError("resolver must be callable")

    policy_name: str | None = None
    if default_policy is not None:
        policy_name = str(default_policy).strip()
        if not policy_name:
            raise NotifyConfigError("default_policy must be a non-empty string when provided")

    alias_names: list[str] = []
    for alias in aliases:
        alias_name = _normalize_name(alias, field="alias")
        if alias_name == tool_name:
            raise NotifyConfigError(f"alias '{alias_name}' cannot equal tool name '{tool_name}'")
        if alias_name in _TOOL_ALIASES or alias_name in _TOOL_RESOLVERS:
            raise NotifyConfigError(f"alias '{alias_name}' is already registered")
        alias_names.append(alias_name)

    _TOOL_RESOLVERS[tool_name] = ToolEventsSourceResolver(resolver=resolver, default_policy=policy_name)
    for alias_name in alias_names:
        _TOOL_ALIASES[alias_name] = tool_name


def normalize_tool_name(tool: str | None) -> str | None:
    if tool is None:
        return None
    value = _normalize_name(tool, field="tool")
    return _TOOL_ALIASES.get(value, value)


def resolve_tool_events_path(*, tool: str, config: Path) -> tuple[Path, str | None]:
    tool_name = normalize_tool_name(tool)
    if tool_name is None:
        raise NotifyConfigError("tool must be a non-empty string when provided")
    resolver = _TOOL_RESOLVERS.get(tool_name)
    if resolver is None:
        allowed = ", ".join(sorted(_TOOL_RESOLVERS))
        raise NotifyConfigError(f"unsupported tool '{tool}'. Supported values: {allowed}")
    events_path = resolver.resolver(config)
    if not isinstance(events_path, Path):
        events_path = Path(events_path)
    return events_path.resolve(), resolver.default_policy


register_builtin_tool_events_sources(register_tool_events_source)
