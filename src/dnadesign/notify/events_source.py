"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/events_source.py

Tool config resolvers for expected USR events log destinations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from .errors import NotifyConfigError


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


def _resolve_densegen_events_from_config(config_path: Path) -> Path:
    try:
        from dnadesign.densegen.src.config import load_config, resolve_outputs_scoped_path, resolve_run_root
    except Exception as exc:
        raise NotifyConfigError(f"failed to load DenseGen resolver dependencies: {exc}") from exc

    cfg_path = config_path.expanduser().resolve()
    if not cfg_path.exists():
        raise NotifyConfigError(f"tool config not found: {cfg_path}")
    if not cfg_path.is_file():
        raise NotifyConfigError(f"tool config is not a file: {cfg_path}")

    try:
        loaded = load_config(cfg_path)
    except Exception as exc:
        raise NotifyConfigError(f"failed to load DenseGen config '{cfg_path}': {exc}") from exc
    cfg = loaded.root.densegen
    out_cfg = cfg.output
    usr_cfg = out_cfg.usr
    if "usr" not in out_cfg.targets or usr_cfg is None:
        raise NotifyConfigError("DenseGen config must enable output.targets 'usr' with output.usr configured")
    dataset = str(usr_cfg.dataset).strip()
    if not dataset:
        raise NotifyConfigError("DenseGen config output.usr.dataset must be a non-empty string")

    try:
        run_root = resolve_run_root(loaded.path, cfg.run.root)
        usr_root = resolve_outputs_scoped_path(loaded.path, run_root, usr_cfg.root, label="output.usr.root")
    except Exception as exc:
        raise NotifyConfigError(f"failed resolving DenseGen USR destination from '{cfg_path}': {exc}") from exc
    return (usr_root / dataset / ".events.log").resolve()


def _infer_usr_root_from_env() -> Path | None:
    env = str(os.environ.get("DNADESIGN_USR_ROOT", "")).strip()
    if not env:
        return None
    return Path(env).expanduser().resolve()


def _resolve_infer_evo2_events_from_config(config_path: Path) -> Path:
    cfg_path = config_path.expanduser().resolve()
    if not cfg_path.exists():
        raise NotifyConfigError(f"tool config not found: {cfg_path}")
    if not cfg_path.is_file():
        raise NotifyConfigError(f"tool config is not a file: {cfg_path}")
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise NotifyConfigError(f"failed to parse infer config '{cfg_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise NotifyConfigError(f"infer config must be a YAML mapping at top-level: {cfg_path}")
    jobs = raw.get("jobs")
    if not isinstance(jobs, list):
        raise NotifyConfigError(f"infer config must include a jobs list: {cfg_path}")

    destinations: set[tuple[Path, str]] = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        ingest = job.get("ingest")
        if not isinstance(ingest, dict):
            continue
        source = str(ingest.get("source") or "").strip().lower()
        if source != "usr":
            continue
        io_cfg = job.get("io")
        io = io_cfg if isinstance(io_cfg, dict) else {}
        if not bool(io.get("write_back")):
            continue
        dataset = str(ingest.get("dataset") or "").strip()
        if not dataset:
            raise NotifyConfigError("infer_evo2 resolver requires ingest.dataset for source='usr' jobs")
        root_value = ingest.get("root")
        if root_value is None:
            usr_root = _infer_usr_root_from_env()
            if usr_root is None:
                raise NotifyConfigError(
                    "infer_evo2 resolver requires ingest.root or DNADESIGN_USR_ROOT for source='usr' jobs"
                )
        else:
            root_text = str(root_value).strip()
            if not root_text:
                raise NotifyConfigError("infer_evo2 resolver received empty ingest.root")
            usr_root = Path(root_text).expanduser().resolve()
        destinations.add((usr_root, dataset))

    if not destinations:
        raise NotifyConfigError(
            "infer_evo2 resolver requires at least one job with ingest.source='usr' and io.write_back=true"
        )
    if len(destinations) > 1:
        rendered = ", ".join(sorted(f"{root}/{dataset}" for root, dataset in destinations))
        raise NotifyConfigError(
            f"infer_evo2 resolver found multiple USR destinations in config: {rendered}. "
            "Pass --events explicitly to select one stream."
        )
    usr_root, dataset = next(iter(destinations))
    return (usr_root / dataset / ".events.log").resolve()


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


register_tool_events_source(
    tool="densegen",
    resolver=_resolve_densegen_events_from_config,
    default_policy="densegen",
)
register_tool_events_source(
    tool="infer_evo2",
    resolver=_resolve_infer_evo2_events_from_config,
    default_policy="infer_evo2",
    aliases=("infer-evo2",),
)
