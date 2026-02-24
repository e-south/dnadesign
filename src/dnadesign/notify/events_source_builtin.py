"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/notify/events_source_builtin.py

Built-in tool config resolvers for notify USR events source discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import yaml

from .errors import NotifyConfigError

ToolEventsSourceRegister = Callable[..., None]


def _resolve_densegen_events_from_config(config_path: Path) -> Path:
    try:
        from dnadesign.densegen import load_config, resolve_outputs_scoped_path, resolve_run_root
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


def register_builtin_tool_events_sources(register: ToolEventsSourceRegister) -> None:
    if not callable(register):
        raise TypeError("register must be callable")
    register(
        tool="densegen",
        resolver=_resolve_densegen_events_from_config,
        default_policy="densegen",
    )
    register(
        tool="infer_evo2",
        resolver=_resolve_infer_evo2_events_from_config,
        default_policy="infer_evo2",
        aliases=("infer-evo2",),
    )
